#ifndef NB_MODEL_HH_
#define NB_MODEL_HH_

template <typename T>
struct nb_model_t {
  using Scalar = typename T::Scalar;
  using Index = typename T::Index;
  using Data = T;

  template <typename X>
  using Dense = Eigen::MatrixBase<X>;

  template <typename X>
  using Sparse = Eigen::SparseMatrixBase<X>;

  explicit nb_model_t(const T& y)
      : n(y.rows()),
        m(y.cols()),
        Y(n, m),
        llik_mat(n, m),
        sampled_mat(n, m),
        phi_mat(n, m),
        phi_min(1, m),
        phi_max(1, m),
        rho_mat(n, m),
        evidence_mat(n, m) {
    alloc_memory(y);
    evidence_mat = y.unaryExpr(is_valid_op);
    Y = y.unaryExpr(rm_invalid_op);
    calc_phi_max();
  }

  const T& llik() const { return llik_mat; }
  const Index n;
  const Index m;

  template <typename Derived, typename OtherDerived>
  const T& eval(const Dense<Derived>& eta_mean,
                const Dense<OtherDerived>& eta_var) {
    return _eval(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& sample(const Dense<Derived>& eta_mean,
                  const Dense<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& sample(const Sparse<Derived>& eta_mean,
                  const Sparse<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

 private:
  template <typename Derived>
  void alloc_memory(const Dense<Derived>& y_raw) {
    const auto n = y_raw.rows();
    const auto m = y_raw.cols();
    Y.setZero(n, m);
    llik_mat.setZero(n, m);
    sampled_mat.setZero(n, m);
    phi_mat.setZero(n, m);
    phi_min.setZero(1, m);
    phi_max.setZero(1, m);
    rho_mat.setZero(n, m);
  }

  T Y;
  T llik_mat;
  T sampled_mat;
  T phi_mat;  // over dispersion parameter
  T phi_min;
  T phi_max;
  T rho_mat;
  T eta_temp_mat;
  T evidence_mat;

  // phi < (var / mean - 1) / mean
  void calc_phi_max() {
    T onesN = T::Ones(1, n);
    T y_num = onesN * evidence_mat;

    auto div_op = [](const Scalar& _d, const Scalar& _n) {
      return static_cast<const Scalar>(_d / (_n + 1e-2));
    };

    T y_mean = (onesN * Y).binaryExpr(y_num, div_op);  // 1 x m

    T y_var = (onesN * Y.cwiseProduct(Y)).binaryExpr(y_num, div_op);  // 1 x m
    y_var -= y_mean.cwiseProduct(y_mean);

    auto phi_max_op = [](const auto& _yv, const auto& _ym) {
      if (_yv <= _ym || _ym <= 1e-2) return one_val + small_val;
      Scalar ret = static_cast<const Scalar>((_yv / _ym - one_val) / _ym);
      if (ret < one_val) return one_val;
      return ret;
    };

    auto phi_min_op = [](const auto& _phi_max) {
      return static_cast<const Scalar>(_phi_max * 1e-2);
    };

    phi_max = y_var.binaryExpr(y_mean, phi_max_op);
    phi_min = phi_max.unaryExpr(phi_min_op);
  }

  // lgamma(y + 1/phi)
  template <typename M1, typename M2>
  const T& _eval(const M1& eta_mean, const M2& eta_var) {
    phi_mat = phi_mat.cwiseProduct(eta_var.unaryExpr(phi_op));

    for (Index j = 0; j < m; ++j) {
      phi_mat.col(j) *= (phi_max(j) - phi_min(j));
    }

    for (Index j = 0; j < m; ++j) {
      phi_mat.col(j).array() += phi_min(j);
    }

    llik_mat = Y.binaryExpr(phi_mat, lngam_ratio_op) -
               Y.binaryExpr(eta_mean, y_lnexp1p_op) -
               phi_mat.binaryExpr(eta_mean, phi_lnexp1p_op);

    llik_mat = llik_mat.cwiseProduct(evidence_mat);
    return llik_mat;
  }

  template <typename M1, typename M2>
  const T& _sample(const M1& eta_mean, const M2& eta_var) {
    phi_mat = eta_var.unaryExpr(phi_op);
    rho_mat = eta_mean.binaryExpr(phi_mat, rho_op);
    auto rnb = [this](const Scalar& phi, const Scalar& rho) {
      return rnb_op(phi, rho);
    };
    sampled_mat = phi_mat.binaryExpr(rho_mat, rnb);
    return sampled_mat;
  }

  ////////////////////////////////////////////////////////////////
  // helper functors (start from zero dispersion)
  // phi = phi.max * 1e-2 + (phi.max - phi.max * 1e-2) * sigmoid(x - 4)
  sigmoid_op_t<Scalar> phi_op;

  // ln Gam(Y + 1/phi + 1) - ln Gam(1/phi + 1)
  struct lngam_ratio_op_t {
    inline Scalar operator()(const Scalar& y, const Scalar& ph) const {
      const Scalar ph_safe = ph + small_val;
      return fasterlgamma(y + one_val / ph_safe + one_val) -
             fasterlgamma(one_val / ph_safe + one_val);
    }
  } lngam_ratio_op;

  // y * ln(1 + exp(-x))
  struct y_lnexp1p_op_t {
    inline Scalar operator()(const Scalar& y, const Scalar& x) const {
      return y * ln1p_exp_op(-x);
    }
    log1pExp_op_t<Scalar> ln1p_exp_op;
  } y_lnexp1p_op;

  // (1/phi + 1) * ln(1 + exp(x))
  struct phi_lnexp1p_op_t {
    Scalar operator()(const Scalar& phi, const Scalar& x) const {
      const Scalar phi_safe = phi + small_val;
      return ln1p_exp_op(x) * (one_val + one_val / phi_safe);
    }
    log1pExp_op_t<Scalar> ln1p_exp_op;
  } phi_lnexp1p_op;

  // sigmoid(eta)
  sigmoid_op_t<Scalar> rho_op;

  struct rnb_op_t {
    using nb_param_type =
        typename std::negative_binomial_distribution<>::param_type;
    rnb_op_t() : rng(std::time(0)) {}
    Scalar operator()(const Scalar& phi, const Scalar& rho) {
      return static_cast<Scalar>(
          distrib(rng, nb_param_type(one_val / phi, rho)));
    }
    std::mt19937 rng;
    std::negative_binomial_distribution<> distrib;
  } rnb_op;

  struct is_valid_op_t {
    Scalar operator()(const Scalar& y) const {
      if (std::isfinite(y) && (y + small_val) > zero_val) return one_val;
      return zero_val;
    }
  } is_valid_op;

  struct rm_invalid_op_t {
    Scalar operator()(const Scalar& y) const {
      if (std::isfinite(y) && (y + small_val) > zero_val) return y;
      return zero_val;
    }
  } rm_invalid_op;

  static constexpr Scalar one_val = 1.0;
  static constexpr Scalar small_val = 1e-8;
  static constexpr Scalar zero_val = 0.0;
};

#endif
