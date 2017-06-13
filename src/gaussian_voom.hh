#ifndef GAUSSIAN_VOOM_MODEL_HH_
#define GAUSSIAN_VOOM_MODEL_HH_

////////////////////////////////////////////////////////////////
// E[Y] = eta_mean
// V[Y] = exp(-eta_mean - var[eta_mean]/2) + g(eta_var)
////////////////////////////////////////////////////////////////
template <typename T>
struct gaussian_voom_model_t {
  using Scalar = typename T::Scalar;
  using Index = typename T::Index;
  using Data = T;

  template <typename X>
  using Dense = Eigen::MatrixBase<X>;

  template <typename X>
  using Sparse = Eigen::SparseMatrixBase<X>;

  struct Vmax_t : public check_positive_t<Scalar> {
    explicit Vmax_t(Scalar v) : check_positive_t<Scalar>(v) {}
  };

  struct Vmin_t : public check_positive_t<Scalar> {
    explicit Vmin_t(Scalar v) : check_positive_t<Scalar>(v) {}
  };

  explicit gaussian_voom_model_t(const T& yy, const Vmin_t& Vmin,
                                 const Vmax_t& Vmax)
      : n(yy.rows()),
        m(yy.cols()),
        Y_safe(n, m),
        llik_mat(n, m),
        sampled_mat(n, m),
        mean_mat(n, m),
        var_mat(n, m),
        evidence_mat(n, m),
        var_op(Vmin.val, Vmax.val),
        col_var_op(mean_mat) {
    is_obs_op<T> obs_op;
    evidence_mat = yy.unaryExpr(obs_op);
    remove_missing(yy, Y_safe);
    alloc_memory(Y_safe);
  }

  template <typename Derived, typename OtherDerived>
  const T& eval(const Dense<Derived>& eta_mean,
                const Dense<OtherDerived>& eta_var) {
    return _eval(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& eval(const Sparse<Derived>& eta_mean,
                const Sparse<OtherDerived>& eta_var) {
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

  const T& llik() const { return llik_mat; }

  const Index n;
  const Index m;
  const T Y_safe;

 private:
  template <typename Derived>
  void alloc_memory(const Dense<Derived>& yy) {
    llik_mat.setZero();
    sampled_mat.setZero();
    mean_mat.setZero();
    var_mat.setZero();
  }

  template <typename Derived>
  void alloc_memory(const Sparse<Derived>& yy) {
    initialize(yy, llik_mat, 0.0);
    initialize(yy, sampled_mat, 0.0);
    initialize(yy, mean_mat, 0.0);
    initialize(yy, var_mat, 0.0);
  }

  // var[Y] = exp(-eta_mean - var[eta_mean]/2) + g(eta_var)
  // llik = -0.5 * var[Y] - 0.5 * (y - mu)^2 / var[Y]
  template <typename M1, typename M2>
  const T& _eval(const M1& eta_mean, const M2& eta_var) {
    mean_mat = eta_mean;                  // E[Y] = eta_mean
    const T& obs_var_vec = col_var_op();  // ncol x 1

    // V[i,g] = phi(eta_var[i,g]) + exp(-0.5 var(mean[,g]) - mean[i,g])
    for (Index g = 0; g < m; ++g) {
      const Scalar v0 = obs_var_vec(g);  // v0 = 0 is okay
      var_mat.col(g).setConstant(-v0 * half_val);
    }
    var_mat -= mean_mat;
    var_mat = var_mat.unaryExpr(exp_op);
    var_mat += eta_var.unaryExpr(var_op);

    llik_mat =
        -half_val * var_mat.unaryExpr(log_op) -
        half_val * Y_safe.binaryExpr(mean_mat, mean_op).cwiseQuotient(var_mat);
    llik_mat = llik_mat.cwiseProduct(evidence_mat);
    return llik_mat;
  }

  template <typename M1, typename M2>
  const T& _sample(const M1& eta_mean, const M2& eta_var) {
    auto rnorm = [this](const Scalar& mu_val, const Scalar& var_val) {
      return rnorm_op(mu_val, var_val);
    };
    mean_mat = eta_mean;

    const T& obs_var_vec = col_var_op();  // ncol x 1

    // V[i,g] = phi(eta_var[i,g]) + exp(-0.5 var(mean[,g]) - mean[i,g])
    for (Index g = 0; g < m; ++g) {
      const Scalar v0 = obs_var_vec(g);  // v0 = 0 is okay
      var_mat.col(g).setConstant(-v0 * half_val);
    }
    var_mat -= mean_mat;
    var_mat = var_mat.unaryExpr(exp_op);
    var_mat += eta_var.unaryExpr(var_op);

    sampled_mat = eta_mean.binaryExpr(var_mat, rnorm);
    return sampled_mat;
  }

  T llik_mat;
  T sampled_mat;
  T mean_mat;
  T var_mat;
  T evidence_mat;

  // variance  ~ (Vmax - Vmin) * sigmoid(eta_var) + Vmin
  struct var_op_t {
    explicit var_op_t(const Scalar vmin, const Scalar vmax)
        : Vmin(vmin), Vmax(vmax), sgm_op() {}
    const Scalar operator()(const Scalar& x) const {
      return (Vmax - Vmin) * sgm_op(x) + Vmin;
    }
    const Scalar Vmin;
    const Scalar Vmax;
    const sigmoid_op_t<Scalar> sgm_op;
    const exp_op_t<Scalar> exp_op;
  } var_op;

  struct dist_func_t {
    Scalar operator()(const Scalar& x, const Scalar& y) const {
      return (x - y) * (x - y);
    }
  } mean_op;

  log_op_t<Scalar> log_op;

  exp_op_t<Scalar> exp_op;

  struct rnorm_op_t {
    rnorm_op_t() : rng(std::time(0)) {}
    Scalar operator()(const Scalar& mu_val, const Scalar& var_val) {
      return distrib(rng) * std::sqrt(var_val) + mu_val;
    }
    std::mt19937 rng;
    std::normal_distribution<Scalar> distrib;
  } rnorm_op;

  column_var_op_t<T> col_var_op;

  static constexpr Scalar small_val = 1e-8;
  static constexpr Scalar half_val = 0.5;
};

#endif
