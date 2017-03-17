#include "adam.hh"
#include "param_check.hh"
#include "rcpp_util.hh"
#include "mathutil.hh"
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/SparseCore>
#include <memory>

#ifndef PARAMETERS_HH_
#define PARAMETERS_HH_

template <typename T>
struct param_traits {
  typedef typename T::data_t Matrix;
  typedef typename T::scalar_t Scalar;
  typedef typename T::index_t Index;
  typedef typename T::grad_adam_t Adam;
  typedef typename T::sgd_tag Sgd;
  typedef typename T::sparsity_tag Sparsity;
};

template <typename T>
using sgd_tag = typename param_traits<T>::Sgd;

template <typename T>
using sparsity_tag = typename param_traits<T>::Sparsity;

struct tag_param_dummy {}; // dummy parameter
struct tag_param_slab {};  // q(theta[j,g]) ~ N(beta[j,g], 1/gamma[j,g])
struct tag_param_spike_slab {
};  // q(theta[j,g]) ~ alpha[j,g] * N(beta[j,g], 1/gamma[j,g]) + (1-alpha[j,g]) * delta(theta)
struct tag_param_col_spike_slab {
};  // q(theta[j,g]) ~ alpha[g] * N(beta[j,g], 1/gamma[g]) + (1-alpha[g]) * delta(theta)
struct tag_param_row_spike_slab {
};  // q(theta[j,g]) ~ alpha[j] * N(beta[j,g], 1/gamma[j]) + (1-alpha[j]) * delta(theta)

struct tag_param_sparse {};
struct tag_param_dense {};

////////////////////////////////////////////////////////////////
// include implementations
#include "param_dummy.hh"
#include "param_slab.hh"
#include "param_spike_slab.hh"
#include "param_row_spike_slab.hh"
#include "param_col_spike_slab.hh"

template <typename Derived>
using SparseDeriv = Eigen::SparseMatrixBase<Derived>;

template <typename Derived>
using DenseDeriv = Eigen::MatrixBase<Derived>;

////////////////////////////////////////////////////////////////
// dispatch functions for SGD evaluation
template <typename P, typename D1, typename D2, typename D3>
void eval_param_sgd(P& p, const Eigen::MatrixBase<D1>& g1, const Eigen::MatrixBase<D2>& g2,
                    const Eigen::MatrixBase<D3>& nobs) {
  safe_eval_param_sgd(p, g1, g2, nobs, sparsity_tag<P>());
}

template <typename P, typename D1, typename D2, typename D3>
void eval_param_sgd(P& p, const Eigen::SparseMatrixBase<D1>& g1, const Eigen::SparseMatrixBase<D2>& g2,
                    const Eigen::SparseMatrixBase<D3>& nobs) {
  safe_eval_param_sgd(p, g1, g2, nobs, sparsity_tag<P>());
}

// check to match sparsity of the parameter and gradient
template <typename Parameter, typename Deriv1, typename Deriv2, typename Deriv3>
void safe_eval_param_sgd(Parameter& P, const Eigen::MatrixBase<Deriv1>& G1, const Eigen::MatrixBase<Deriv2>& G2,
                         const Eigen::MatrixBase<Deriv3>& Nobs, const tag_param_dense) {
  impl_eval_param_sgd(P, G1.derived(), G2.derived(), Nobs.derived(), sgd_tag<Parameter>());
}

template <typename Parameter, typename Deriv1, typename Deriv2, typename Deriv3>
void safe_eval_param_sgd(Parameter& P, const Eigen::SparseMatrixBase<Deriv1>& G1,
                         const Eigen::SparseMatrixBase<Deriv2>& G2, const Eigen::SparseMatrixBase<Deriv3>& Nobs,
                         const tag_param_sparse) {
  impl_eval_param_sgd(P, G1.derived(), G2.derived(), Nobs.derived(), sgd_tag<Parameter>());
}

////////////////////////////////////////////////////////////////
// dispatch functions for SGD evaluation
template <typename P, typename D1, typename D2, typename D3>
void eval_hyperparam_sgd(P& p, const Eigen::MatrixBase<D1>& g1, const Eigen::MatrixBase<D2>& g2,
                         const Eigen::MatrixBase<D3>& nobs) {
  safe_eval_hyperparam_sgd(p, g1, g2, nobs, sparsity_tag<P>());
}

template <typename P, typename D1, typename D2, typename D3>
void eval_hyperparam_sgd(P& p, const Eigen::SparseMatrixBase<D1>& g1, const Eigen::SparseMatrixBase<D2>& g2,
                         const Eigen::SparseMatrixBase<D3>& nobs) {
  safe_eval_hyperparam_sgd(p, g1, g2, nobs, sparsity_tag<P>());
}

// check to match sparsity of the parameter and gradient
template <typename Parameter, typename Deriv1, typename Deriv2, typename Deriv3>
void safe_eval_hyperparam_sgd(Parameter& P, const Eigen::MatrixBase<Deriv1>& G1, const Eigen::MatrixBase<Deriv2>& G2,
                              const Eigen::MatrixBase<Deriv3>& Nobs, const tag_param_dense) {
  impl_eval_hyperparam_sgd(P, G1.derived(), G2.derived(), Nobs.derived(), sgd_tag<Parameter>());
}

template <typename Parameter, typename Deriv1, typename Deriv2, typename Deriv3>
void safe_eval_hyperparam_sgd(Parameter& P, const Eigen::SparseMatrixBase<Deriv1>& G1,
                              const Eigen::SparseMatrixBase<Deriv2>& G2, const Eigen::SparseMatrixBase<Deriv3>& Nobs,
                              const tag_param_sparse) {
  impl_eval_hyperparam_sgd(P, G1.derived(), G2.derived(), Nobs.derived(), sgd_tag<Parameter>());
}

////////////////////////////////////////////////////////////////
// dispatch functions for initialization
template <typename Parameter>
void initialize_param(Parameter& P) {
  impl_initialize_param(P, sgd_tag<Parameter>());
}

// dispatch functions for update
template <typename Parameter, typename Scalar>
void update_param_sgd(Parameter& P, const Scalar rate) {
  impl_update_param_sgd(P, rate, sgd_tag<Parameter>());
}

template <typename Parameter, typename Scalar>
void update_hyperparam_sgd(Parameter& P, const Scalar rate) {
  impl_update_hyperparam_sgd(P, rate, sgd_tag<Parameter>());
}

// dispatch functions for resolution
template <typename Parameter>
void resolve_param(Parameter& P) {
  impl_resolve_param(P, sgd_tag<Parameter>());
}

// dispatch functions for resolution
template <typename Parameter>
void resolve_hyperparam(Parameter& P) {
  impl_resolve_hyperparam(P, sgd_tag<Parameter>());
}

// dispatch functions for perturbation
template <typename Parameter, typename Scalar>
void perturb_param(Parameter& P, const Scalar sd) {
  impl_perturb_param(P, sd, sgd_tag<Parameter>());
}

template <typename Parameter>
void check_nan_param(Parameter& P, std::string msg) {
  std::cerr << msg;
  impl_check_nan_param(P, sgd_tag<Parameter>());
  std::cerr << " -> ok" << std::endl;
}

////////////////////////////////////////////////////////////////
template <typename Parameter>
auto log_odds_param(Parameter& P) {
  return impl_log_odds_param(P, sgd_tag<Parameter>());
}

template <typename Parameter>
const auto& mean_param(Parameter& P) {
  return impl_mean_param(P, sgd_tag<Parameter>());
}

template <typename Parameter>
const auto& var_param(Parameter& P) {
  return impl_var_param(P, sgd_tag<Parameter>());
}

template <typename Parameter>
void write_param(Parameter& P, const std::string hdr, const std::string gz) {
  impl_write_param(P, hdr, gz, sgd_tag<Parameter>());
}

#endif
