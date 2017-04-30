// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "convergence.hh"
#include "dummy.hh"
#include "factorization.hh"
#include "gaussian.hh"
#include "gaussian_voom.hh"
#include "options.hh"
#include "parameters.hh"
#include "rcpp_util.hh"
#include "regression.hh"
#include "regression_factored.hh"
#include "residual.hh"
#include "sgvb_inference.hh"
#include "shared_effect.hh"
#include <memory>

#ifndef RCPP_FQT_HH_
#define RCPP_FQT_HH_

using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor>;
using Scalar = Mat::Scalar;
using Index = Mat::Index;

struct m_gaussian_tag {};
struct m_gaussian_voom_tag {};

template <typename Mat, typename ModelT> struct impl_model_maker_t;

template <typename Mat> struct impl_model_maker_t<Mat, m_gaussian_tag> {
  using Scalar = typename Mat::Scalar;
  using model_type = gaussian_model_t<Mat>;

  std::shared_ptr<model_type> operator()(const Mat &Y) {
    calc_stat_t<Scalar> stat_op;
    visit(Y, stat_op);
    const Scalar vmax = stat_op.var();
    const Scalar vmin = 1e-4 * vmax;
    TLOG("Gaussian model : Vmax " << vmax << ", Vmin " << vmin);
    return std::make_shared<model_type>(Y, typename model_type::Vmin_t(vmin),
                                        typename model_type::Vmax_t(vmax));
  }
};

template <typename Mat> struct impl_model_maker_t<Mat, m_gaussian_voom_tag> {
  using Scalar = typename Mat::Scalar;
  using model_type = gaussian_voom_model_t<Mat>;

  std::shared_ptr<model_type> operator()(const Mat &Y) {
    calc_stat_t<Scalar> stat_op;
    visit(Y, stat_op);
    const Scalar vmax = stat_op.var();
    const Scalar vmin = 1e-4 * vmax;
    TLOG("Gaussian voom model : Vmax " << vmax << ", Vmin " << vmin);
    return std::make_shared<model_type>(Y, typename model_type::Vmin_t(vmin),
                                        typename model_type::Vmax_t(vmax));
  }
};

template <typename T, typename Mat> auto make_model(const Mat &Y) {
  impl_model_maker_t<Mat, T> maker;
  return maker(Y);
}

////////////////////////////////////////////////////////////////
template <typename Derived, typename OtherDerived>
void summarize_llik(const Eigen::MatrixBase<Derived> &llik,
                    Eigen::MatrixBase<OtherDerived> &llik_trace) {
  llik_trace.resize(llik.rows(), 1);
  llik_trace.derived() = llik.rowwise().mean();
}

////////////////////////////////////////////////////////////////
void set_options_from_list(const Rcpp::List &_list, options_t &opt) {
  if (_list.containsElementNamed("tau.lb"))
    opt.TAU_LODDS_LB = Rcpp::as<Scalar>(_list["tau.lb"]);
  if (_list.containsElementNamed("tau.ub"))
    opt.TAU_LODDS_UB = Rcpp::as<Scalar>(_list["tau.ub"]);
  if (_list.containsElementNamed("pi.lb"))
    opt.PI_LODDS_LB = Rcpp::as<Scalar>(_list["pi.lb"]);
  if (_list.containsElementNamed("pi.ub"))
    opt.PI_LODDS_UB = Rcpp::as<Scalar>(_list["pi.ub"]);
  if (_list.containsElementNamed("tol"))
    opt.VBTOL = Rcpp::as<Scalar>(_list["tol"]);
  if (_list.containsElementNamed("k"))
    opt.K = Rcpp::as<Scalar>(_list["k"]);
  if (_list.containsElementNamed("K"))
    opt.K = Rcpp::as<Scalar>(_list["K"]);
  if (_list.containsElementNamed("gammax"))
    opt.GAMMAX = Rcpp::as<Scalar>(_list["gammax"]);
  if (_list.containsElementNamed("decay"))
    opt.DECAY = Rcpp::as<Scalar>(_list["decay"]);
  if (_list.containsElementNamed("rate"))
    opt.RATE0 = Rcpp::as<Scalar>(_list["rate"]);

  if (_list.containsElementNamed("adam.rate.m"))
    opt.RATE_M = Rcpp::as<Scalar>(_list["adam.rate.m"]);

  if (_list.containsElementNamed("adam.rate.v"))
    opt.RATE_V = Rcpp::as<Scalar>(_list["adam.rate.v"]);

  if (_list.containsElementNamed("rseed"))
    opt.RSEED = Rcpp::as<Index>(_list["rseed"]);

  if (_list.containsElementNamed("vbiter"))
    opt.VBITER = Rcpp::as<Index>(_list["vbiter"]);
  if (_list.containsElementNamed("verbose"))
    opt.VERBOSE = Rcpp::as<bool>(_list["verbose"]);
  if (_list.containsElementNamed("out.residual"))
    opt.OUT_RESID = Rcpp::as<bool>(_list["out.residual"]);
  // if (_list.containsElementNamed("model"))
  //   opt.MODEL_NAME = Rcpp::as<std::string>(_list["model"]);
}

////////////////////////////////////////////////////////////////
Rcpp::List rcpp_fqtl_adj_list(const Rcpp::NumericVector &d1_loc,
                              const Rcpp::NumericVector &d2_start_loc,
                              const Rcpp::NumericVector &d2_end_loc,
                              const double cis_window) {
  const auto n1 = d1_loc.size();
  const auto n2 = d2_start_loc.size();

  if (d2_start_loc.size() != d2_end_loc.size()) {
    ELOG("start and end location vectors have different size");
    return Rcpp::List::create();
  }

  std::vector<int> left;
  std::vector<int> right;

  for (auto i = 0u; i < n1; ++i) {
    const double d1 = d1_loc.at(i);
    for (auto j = 0u; j < n2; ++j) {
      const double d2_start = d2_start_loc[j];
      const double d2_end = d2_end_loc[j];
      if (d2_start > d2_end)
        continue;
      if (d1 >= (d2_start - cis_window) && d1 <= (d2_end + cis_window)) {
        left.push_back(i + 1);
        right.push_back(j + 1);
      }
    }
  }

  return Rcpp::List::create(Rcpp::_["d1"] = Rcpp::wrap(left),
                            Rcpp::_["d2"] = Rcpp::wrap(right));
}

////////////////////////////////////////////////////////////////
template <typename T> Rcpp::List param_rcpp_list(const T &param) {
  return impl_param_rcpp_list(param, sgd_tag<T>());
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_spike_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param,
                                const tag_param_col_spike_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_col_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param));
}

#endif
