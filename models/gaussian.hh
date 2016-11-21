#include "engine/mathutil.hh"
#include "utils/eigen_util.hh"
#include "utils/param_check.hh"

#ifndef GAUSSIAN_MODEL_HH_
#define GAUSSIAN_MODEL_HH_

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
      : Y{yy.unaryExpr([](const Scalar& y) { return static_cast<Scalar>(std::isfinite(y) ? y : 0.0); })},
        n{Y.rows()},
        m{Y.cols()},
        llik_mat{n, m},
        sampled_mat{n, m},
        mean_mat{n, m},
        var_mat{n, m},
        var_op{Vmin.val, Vmax.val} {
    alloc_memory(Y);
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

  const T Y;
  const Index n;
  const Index m;

 private:
  template <typename Derived>
  void alloc_memory(const Dense<Derived>& Y) {
    llik_mat.setZero();
    sampled_mat.setZero();
    mean_mat.setZero();
    var_mat.setZero();
  }

  template <typename Derived>
  void alloc_memory(const Sparse<Derived>& Y) {
    initialize(Y, llik_mat, 0.0);
    initialize(Y, sampled_mat, 0.0);
    initialize(Y, mean_mat, 0.0);
    initialize(Y, var_mat, 0.0);
  }

  // -0.5 * ln Var - 0.5 * (y - mu)^2 / Var
  template <typename M1, typename M2>
  const T& _eval(const M1& eta_mean, const M2& eta_var) {
    var_mat = eta_var.unaryExpr(var_op);
    mean_mat = eta_mean;
    llik_mat = -0.5 * var_mat.unaryExpr(log_op) - 0.5 * Y.binaryExpr(mean_mat, mean_op).cwiseQuotient(var_mat);
    return llik_mat;
  }

  template <typename M1, typename M2>
  const T& _sample(const M1& eta_mean, const M2& eta_var) {
    auto rnorm = [this](const Scalar& mu_val, const Scalar& var_val) { return rnorm_op(mu_val, var_val); };
    var_mat = eta_var.unaryExpr(var_op);
    mean_mat = eta_mean;
    sampled_mat = mean_mat.binaryExpr(var_mat, rnorm);
    return sampled_mat;
  }

  T llik_mat;
  T sampled_mat;
  T mean_mat;
  T var_mat;

  // mean      ~ eta_mean
  // variance  ~ Vmax * sigmoid(eta_var) + Vmin
  struct var_op_t {
    explicit var_op_t(const Scalar vmin, const Scalar vmax) : Vmin(vmin), Vmax(vmax), sgm_op() {}
    const Scalar operator()(const Scalar& x) const { return Vmax * sgm_op(x) + Vmin; }
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
    Scalar operator()(const Scalar& mu_val, const Scalar& var_val) {
      return distrib(rng) * std::sqrt(var_val) + mu_val;
    }
    std::mt19937 rng;
    std::normal_distribution<Scalar> distrib;
  } rnorm_op;
};

#endif
