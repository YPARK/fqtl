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

  struct alpha_min_t : public check_positive_t<Scalar> {
    explicit alpha_min_t(Scalar v) : check_positive_t<Scalar>(v) {}
  };

  explicit nb_model_t(const T& y, const alpha_min_t& _amin)
      : n(y.rows()),
        m(y.cols()),
        Y(n, m),
        llik_mat(n, m),
        sampled_mat(n, m),
        rho_mat(n, m),
        alpha_mat(n, m),
        evidence_mat(n, m),
        alpha_op(_amin.val) {
    alloc_memory(y);
    evidence_mat = y.unaryExpr(is_valid_op);
    Y = y.unaryExpr(rm_invalid_op);
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
    alpha_mat.setZero(n, m);
    rho_mat.setZero(n, m);
  }

  T Y;
  T llik_mat;
  T sampled_mat;
  T rho_mat;

  T alpha_mat;  // inverse of dispersion parameter

  T eta_temp_mat;
  T evidence_mat;

  // alpha = alpha.min + relu(-eta_var)
  // beta = exp(-eta_mean)

  // llik =
  // ln gamma(y + alpha) - ln gamma(alpha)
  // - alpha * ln(1 + exp(eta.mean))
  // - y * ln(1 + exp(-eta.mean))

  template <typename M1, typename M2>
  const T& _eval(const M1& eta_mean, const M2& eta_var) {
    alpha_mat = eta_var.unaryExpr(alpha_op);

    llik_mat = Y.binaryExpr(alpha_mat, lngam_ratio_op) -
               Y.binaryExpr(eta_mean, y_lnexp1p_op) -
               alpha_mat.binaryExpr(eta_mean, alpha_lnexp1p_op);

    llik_mat = llik_mat.cwiseProduct(evidence_mat);
    return llik_mat;
  }

  template <typename M1, typename M2>
  const T& _sample(const M1& eta_mean, const M2& eta_var) {
    alpha_mat = eta_var.unaryExpr(alpha_op);
    rho_mat = eta_mean.binaryExpr(rho_op);
    auto rnb = [this](const Scalar& alpha, const Scalar& rho) {
      return rnb_op(alpha, rho);
    };

    sampled_mat = alpha_mat.binaryExpr(rho_mat, rnb);
    return sampled_mat;
  }

  ////////////////////////////////////////////////////////////////
  struct alpha_op_t {
    explicit alpha_op_t(const Scalar _amin)
        : alpha_min(_amin) {}

    inline Scalar operator()(const Scalar& x) const {
      return alpha_min + ln1p_exp_op(-x);
    }

    const Scalar alpha_min;
    log1pExp_op_t<Scalar> ln1p_exp_op;
  } alpha_op;

  // ln Gam(Y + alpha) - ln Gam(alpha)
  struct lngam_ratio_op_t {
    inline Scalar operator()(const Scalar& y, const Scalar& aa) const {
      return fasterlgamma(y + aa) - fasterlgamma(aa);
    }
  } lngam_ratio_op;

  // y * ln(1 + exp(-x))
  struct y_lnexp1p_op_t {
    inline Scalar operator()(const Scalar& y, const Scalar& x) const {
      return y * ln1p_exp_op(-x);
    }
    log1pExp_op_t<Scalar> ln1p_exp_op;
  } y_lnexp1p_op;

  // aa * ln(1 + exp(x))
  struct alpha_lnexp1p_op_t {
    Scalar operator()(const Scalar& aa, const Scalar& x) const {
      return ln1p_exp_op(x) * aa;
    }
    log1pExp_op_t<Scalar> ln1p_exp_op;
  } alpha_lnexp1p_op;

  // sigmoid(eta)
  sigmoid_op_t<Scalar> rho_op;

  struct rnb_op_t {
    using nb_param_type =
        typename std::negative_binomial_distribution<>::param_type;
    rnb_op_t() : rng(std::time(0)) {}
    Scalar operator()(const Scalar& aa, const Scalar& rho) {
      return static_cast<Scalar>(distrib(rng, nb_param_type(aa, rho)));
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
