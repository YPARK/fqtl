#include "engine/mathutil.hh"
#include "utils/eigen_util.hh"
#include "utils/param_check.hh"

#ifndef BETA_MODEL_HH_
#define BETA_MODEL_HH_

template <typename T>
struct beta_model_t {
  using Scalar = typename T::Scalar;
  using Index = typename T::Index;
  using Data = T;

  struct beta_hyper_t : public check_positive_t<Scalar> {
    explicit beta_hyper_t(Scalar v) : check_positive_t<Scalar>(v) {}
  };
  struct beta_maxprec_t : public check_positive_t<Scalar> {
    explicit beta_maxprec_t(Scalar v) : check_positive_t<Scalar>(v) {}
  };

  template <typename X>
  using Dense = Eigen::MatrixBase<X>;

  template <typename X>
  using Sparse = Eigen::SparseMatrixBase<X>;

  explicit beta_model_t(const T& Y, const beta_hyper_t& beta_tol, const beta_maxprec_t& beta_max_prec)
      : n(Y.rows()),
        m(Y.cols()),
        llik_mat(n, m),
        sampled_mat(n, m),
        mu(n, m),
        phi(n, m),
        lbeta_op(beta_tol.val),
        rbeta_op(beta_tol.val),
        phi_op(beta_max_prec.val) {
    alloc_memory(Y);
    beta_model_t<T>::preprocess(Y, logitY, log1pMinusY);
  }

  const T& llik() const { return llik_mat; }
  const Index n;
  const Index m;

  template <typename Derived, typename OtherDerived>
  const T& eval(const Dense<Derived>& eta_mean, const Dense<OtherDerived>& eta_var) {
    return _eval(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& eval(const Sparse<Derived>& eta_mean, const Sparse<OtherDerived>& eta_var) {
    return _eval(eta_mean.derived(), eta_var.derived());
  }

  // x ~ Beta(a, b)
  // x1 ~ Gamma(a, 1)
  // X2 ~ Gamma(b, 1)
  // x = x1 / (x1 + x2)
  template <typename Derived, typename OtherDerived>
  const T& sample(const Dense<Derived>& eta_mean, const Dense<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& sample(const Sparse<Derived>& eta_mean, const Sparse<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

 private:
  T logitY;
  T log1pMinusY;

  T llik_mat;
  T sampled_mat;
  T mu;
  T phi;

  std::mt19937 rng;
  std::gamma_distribution<Scalar> rgammaA;
  std::gamma_distribution<Scalar> rgammaB;

  template <typename Derived>
  void alloc_memory(const Dense<Derived>& Y) {
    llik_mat.setZero();
    sampled_mat.setZero();
    mu.setZero();
    phi.setZero();
  }

  template <typename Derived>
  void alloc_memory(const Sparse<Derived>& Y) {
    initialize(Y, llik_mat, 0.0);
    initialize(Y, sampled_mat, 0.0);
    initialize(Y, mu, 0.0);
    initialize(Y, phi, 0.0);
  }

  template <typename Derived>
  static void preprocess(const Dense<Derived>& y_raw, Dense<Derived>& logit_y, Dense<Derived>& log1pMinus_y) {
    _preprocess(y_raw.derived(), logit_y.derived(), log1pMinus_y.derived());
  }

  template <typename Derived>
  static void preprocess(const Sparse<Derived>& y_raw, Sparse<Derived>& logit_y, Sparse<Derived>& log1pMinus_y) {
    _preprocess(y_raw.derived(), logit_y.derived(), log1pMinus_y.derived());
  }
  ////////////////////////////////////////////////////////////////
  // ln Gamma(phi) - ln Gamma(mu*phi) - ln Gamma(phi - mu*phi)
  // + mu*phi * ln(y/(1-y)) + phi * ln(1-y)
  template <typename M1, typename M2>
  const T& _eval(const M1& eta_mean, const M2& eta_var) {
    mu = eta_mean.unaryExpr(sigmoid_op);
    phi = eta_var.unaryExpr(phi_op);
    llik_mat = mu.binaryExpr(phi, lbeta_op) + logitY.cwiseProduct(mu).cwiseProduct(phi) + log1pMinusY.cwiseProduct(phi);
    return llik_mat;
  }

  template <typename M1, typename M2>
  const T& _sample(const M1& eta_mean, const M2& eta_var) {
    mu = eta_mean.unaryExpr(sigmoid_op);
    phi = eta_var.unaryExpr(phi_op);
    auto rbeta = [this](const Scalar& m, const Scalar& p) { return rbeta_op(m, p); };
    sampled_mat = mu.binaryExpr(phi, rbeta);
    return sampled_mat;
  }

  template <typename M>
  static void _preprocess(const M& y_raw, M& logit_y, M& log1pMinus_y) {
    auto clamp_op = [](const Scalar& _y) {
      Scalar y = _y;
      if (y < sgm_lb) y = sgm_lb;
      if (y > sgm_ub) y = sgm_ub;
      return y;
    };

    auto logit_op = [&clamp_op](const Scalar& y) {
      if (!std::isfinite(y)) return (Scalar)0.;
      const Scalar y_safe = clamp_op(y);
      return fasterlog(y_safe) - std::log1p(-y_safe);
    };

    auto log1pMinus_op = [&clamp_op](const Scalar& y) {
      if (!std::isfinite(y)) return (Scalar)0.;
      const Scalar y_safe = clamp_op(y);
      return std::log1p(-y_safe);
    };

    const auto n = y_raw.rows();
    const auto m = y_raw.cols();

    logit_y.resize(n, m);
    log1pMinus_y.resize(n, m);

    logit_y = y_raw.unaryExpr(logit_op);
    log1pMinus_y = y_raw.unaryExpr(log1pMinus_op);
  }

  static constexpr Scalar sgm_lb = 1e-4;
  static constexpr Scalar sgm_ub = 1.0 - 1e-4;

  ////////////////////////////////////////////////////////////////
  // helper functors
  struct lbeta_op_t {
    explicit lbeta_op_t(const Scalar tol) : TOL(tol) {}
    Scalar operator()(const Scalar& mu_val, const Scalar& phi_val) const {
      const Scalar a = mu_val * phi_val + TOL;
      const Scalar b = (one_val - mu_val) * phi_val + TOL;
      return fasterlgamma(a + b) - fasterlgamma(a) - fasterlgamma(b);
      // This could be slow
      // return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
    }
    const Scalar TOL;
  } lbeta_op;

  struct rbeta_op_t {
    explicit rbeta_op_t(const Scalar tol) : TOL(tol) {}

    using gamma_param_type = typename std::gamma_distribution<Scalar>::param_type;

    Scalar operator()(const Scalar& mu_val, const Scalar& phi_val) {
      const Scalar a = rgammaA(rng, gamma_param_type(TOL + mu_val * phi_val, one_val));
      const Scalar b = rgammaB(rng, gamma_param_type(TOL + (one_val - mu_val) * phi_val, one_val));
      return a / (a + b);
    }
    const Scalar TOL;

    std::mt19937 rng;
    std::gamma_distribution<Scalar> rgammaA;
    std::gamma_distribution<Scalar> rgammaB;

  } rbeta_op;

  sigmoid_op_t<Scalar> sigmoid_op;

  struct precision_function_t {
    explicit precision_function_t(const Scalar phimax) : PHIMAX(phimax) {}
    Scalar operator()(const Scalar& eta_var) const { return PHIMAX / (one_val + fasterexp(eta_var)); }
    const Scalar PHIMAX;
  } phi_op;

  static constexpr Scalar one_val = 1.0;
};

#endif
