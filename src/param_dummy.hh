#include "dummy.hh"
#include <Rcpp.h>
#include <RcppCommon.h>
#ifndef PARAM_DUMMY_HH_
#define PARAM_DUMMY_HH_

struct dummy_theta_t {
  typedef float data_t;
  typedef float scalar_t;
  typedef int index_t;
  typedef adam_t<float, scalar_t> grad_adam_t;
  typedef tag_param_dummy sgd_tag;
  typedef tag_param_dense sparsity_tag;

  dummy_mat_t dummy_mat;
};

// clear contents
template <typename Parameter>
void impl_initialize_param(Parameter& P, const tag_param_dummy) {}

// variance is not straightforward
template <typename Parameter>
void impl_resolve_param(Parameter& P, const tag_param_dummy) {}

template <typename Parameter>
void impl_resolve_hyperparam(Parameter& P, const tag_param_dummy) {}

template <typename Parameter, typename Scalar>
void impl_perturb_param(Parameter& P, const Scalar sd, const tag_param_dummy) {}

template <typename Parameter>
const auto& impl_mean_param(Parameter& P, const tag_param_dummy) {
  return P.dummy_mat;
}

template <typename Parameter>
const auto& impl_log_odds_param(Parameter& P, const tag_param_dummy) {
  return P.dummy_mat;
}

template <typename Parameter>
const auto& impl_var_param(Parameter& P, const tag_param_dummy) {
  return P.dummy_mat;
}

template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_param_sgd(Parameter& P, const M1& G1, const M2& G2, const M3& Nobs, const tag_param_dummy) {}

template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_hyperparam_sgd(Parameter& P, const M1& G1, const M2& G2, const M3& Nobs, const tag_param_dummy) {}

// update parameters by calculated stochastic gradient
template <typename Parameter, typename Scalar>
void impl_update_param_sgd(Parameter& P, const Scalar rate, const tag_param_dummy) {}

template <typename Parameter, typename Scalar>
void impl_update_hyperparam_sgd(Parameter& P, const Scalar rate, const tag_param_dummy) {}

template <typename Parameter>
void impl_write_param(Parameter& P, const std::string hdr, const std::string gz, const tag_param_dummy) {}

template <typename Parameter>
void impl_check_nan_param(Parameter& P, const tag_param_dummy) {}

#endif
