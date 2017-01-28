#ifndef GAUSSIAN_MODEL_HH_
#define GAUSSIAN_MODEL_HH_

// E[Y] = eta_mean
// V[Y] = residual variance + vmin + vmax * sigmoid(eta_var)

template <typename T>
struct gaussian_model_t {
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

  explicit gaussian_model_t(const T& yy, const Vmin_t& Vmin, const Vmax_t& Vmax)
      : n(yy.rows()),
        m(yy.cols()),
        Y_safe(n, m),
        llik_mat(n, m),
        sampled_mat(n, m),
        resid_mat(n, m),
        var_mat(n, m),
        resid_var(m, 1),
        onesN(n, 1),
        evidence_mat(n, m),
        var_op(Vmin.val, Vmax.val) {
    is_obs_op<T> obs_op;
    evidence_mat = yy.unaryExpr(obs_op);
    remove_missing(yy, Y_safe);
    alloc_memory(Y_safe);
    onesN.setOnes();
  }

  template <typename Derived, typename OtherDerived>
  const T& eval(const Dense<Derived>& eta_mean, const Dense<OtherDerived>& eta_var) {
    return _eval(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& eval(const Sparse<Derived>& eta_mean, const Sparse<OtherDerived>& eta_var) {
    return _eval(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& sample(const Dense<Derived>& eta_mean, const Dense<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& sample(const Sparse<Derived>& eta_mean, const Sparse<OtherDerived>& eta_var) {
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
    resid_mat.setZero();
    var_mat.setZero();
  }

  template <typename Derived>
  void alloc_memory(const Sparse<Derived>& yy) {
    initialize(yy, llik_mat, 0.0);
    initialize(yy, sampled_mat, 0.0);
    initialize(yy, resid_mat, 0.0);
    initialize(yy, var_mat, 0.0);
  }

  // -0.5 * ln Var - 0.5 * (y - mu)^2 / Var
  template <typename M1, typename M2>
  const T& _eval(const M1& eta_mean, const M2& eta_var) {

    resid_mat = Y_safe - eta_mean;

    // residual variance
    resid_var = resid_mat.cwiseProduct(resid_mat).transpose() * onesN / static_cast<Scalar>(n);

    for (Index g = 0; g < m; ++g) {
      const Scalar v0 = resid_var(g);  // v0 = 0 is okay
      var_mat.col(g).setConstant(v0);
    }

    var_mat += eta_var.unaryExpr(var_op);

    llik_mat = -half_val * var_mat.unaryExpr(log_op);
    llik_mat -= half_val * resid_mat.cwiseProduct(resid_mat).cwiseQuotient(var_mat);
    llik_mat = llik_mat.cwiseProduct(evidence_mat);
    return llik_mat;
  }

  template <typename M1, typename M2>
  const T& _sample(const M1& eta_mean, const M2& eta_var) {
    auto rnorm = [this](const Scalar& mu_val, const Scalar& var_val) { return rnorm_op(mu_val, var_val); };

    resid_mat = Y_safe - eta_mean;

    // residual variance
    resid_var = resid_mat.cwiseProduct(resid_mat).transpose() * onesN / static_cast<Scalar>(n);

    for (Index g = 0; g < m; ++g) {
      const Scalar v0 = resid_var(g);  // v0 = 0 is okay
      var_mat.col(g).setConstant(v0);
    }

    var_mat += eta_var.unaryExpr(var_op);

    sampled_mat = eta_mean.binaryExpr(var_mat, rnorm);
    return sampled_mat;
  }

  T llik_mat;
  T sampled_mat;
  T resid_mat;
  T var_mat;
  T resid_var;
  T onesN;
  T evidence_mat;  

  // mean      ~ eta_mean
  // variance  ~ (Vmax - Vmin) * sigmoid(eta_var) + Vmin + residual variance
  struct var_op_t {
    explicit var_op_t(const Scalar vmin, const Scalar vmax) : Vmin(vmin), Vmax(vmax), sgm_op() {}
    const Scalar operator()(const Scalar& x) const { return (Vmax - Vmin) * sgm_op(x) + Vmin; }
    const Scalar Vmin;
    const Scalar Vmax;
    const sigmoid_op_t<Scalar> sgm_op;
    const exp_op_t<Scalar> exp_op;
  } var_op;

  struct dist_func_t {
    Scalar operator()(const Scalar& x, const Scalar& y) const { return (x - y) * (x - y); }
  } mean_op;

  log_op_t<Scalar> log_op;

  struct rnorm_op_t {
    rnorm_op_t() : rng(std::time(0)) {}
    Scalar operator()(const Scalar& mu_val, const Scalar& var_val) {
      return distrib(rng) * std::sqrt(var_val) + mu_val;
    }
    std::mt19937 rng;
    std::normal_distribution<Scalar> distrib;
  } rnorm_op;

  static constexpr Scalar half_val = 0.5;
};

#endif
