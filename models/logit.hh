#include "engine/mathutil.hh"
#include "utils/eigen_util.hh"
#include "utils/param_check.hh"

#ifndef LOGIT_MODEL_HH_
#define LOGIT_MODEL_HH_

template <typename T>
struct logit_model_t {
  using Scalar = typename T::Scalar;
  using Index = typename T::Index;
  using Data = T;

  template <typename X>
  using Dense = Eigen::MatrixBase<X>;

  template <typename X>
  using Sparse = Eigen::SparseMatrixBase<X>;

  explicit logit_model_t(const T& Y)
      : n(Y.rows()), m(Y.cols()), Y_mat(n, m), llik_mat(n, m), sampled_mat(n, m), runif(0.0, 1.0) {
    alloc_memory(Y);
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

  template <typename Derived, typename OtherDerived>
  const T& sample(const Dense<Derived>& eta_mean, const Dense<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& sample(const Sparse<Derived>& eta_mean, const Sparse<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

 private:
  T Y_mat;
  T llik_mat;
  T sampled_mat;

  std::mt19937 rng;
  std::uniform_real_distribution<Scalar> runif;

  template <typename Derived>
  void alloc_memory(const Dense<Derived>& Y) {
    llik_mat.setZero();
    sampled_mat.setZero();
    Y_mat.setZero();
    copy_matrix(Y, Y_mat);
  }

  template <typename Derived>
  void alloc_memory(const Sparse<Derived>& Y) {
    initialize(Y, llik_mat, 0.0);
    initialize(Y, sampled_mat, 0.0);
    initialize(Y, Y_mat, 0.0);
    copy_matrix(Y, Y_mat);
  }

  ////////////////////////////////////////////////////////////////
  // y * eta_mean - log(1 + exp(eta_mean))
  template <typename M1, typename M2>
  const T& _eval(const M1& eta_mean, const M2& eta_var) {
    llik_mat = Y_mat.cwiseProduct(eta_mean) - eta_mean.unaryExpr(log1pExp);
    return llik_mat;
  }

  template <typename M1, typename M2>
  const T& _sample(const M1& eta_mean, const M2& eta_var) {
    auto rbernoulli = [this](const auto& p) { return runif(rng) < p ? one_val : zero_val; };
    sampled_mat = eta_mean.unaryExpr(sgm);
    sampled_mat = sampled_mat.unaryExpr(rbernoulli);
    return sampled_mat;
  }

  log1pExp_op_t<Scalar> log1pExp;
  sigmoid_op_t<Scalar> sgm;

  static constexpr Scalar sgm_lb = 1e-4;
  static constexpr Scalar sgm_ub = 1.0 - 1e-4;
  static constexpr Scalar one_val = 1.0;
  static constexpr Scalar zero_val = 1.0;
};

#endif
