// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
// [[Rcpp::plugins(openmp)]]
#include <omp.h>

#include <memory>
#include "convergence.hh"
#include "dummy.hh"
#include "factorization.hh"
#include "gaussian.hh"
#include "gaussian_voom.hh"
#include "nb.hh"
#include "options.hh"
#include "parameters.hh"
#include "rcpp_util.hh"
#include "regression.hh"
#include "regression_factored.hh"
#include "residual.hh"
#include "sgvb_inference.hh"
#include "shared_effect.hh"

#ifndef RCPP_FQT_HH_
#define RCPP_FQT_HH_

////////////////////
// Package export //
////////////////////

// Factorization routines

RcppExport SEXP fqtl_rcpp_train_mf(SEXP y, SEXP x_m, SEXP x_v, SEXP opt_mf,
                                   SEXP opt_reg);

RcppExport SEXP fqtl_rcpp_train_mf_cis(SEXP y, SEXP x_m, SEXP a_m, SEXP x_v,
                                       SEXP opt_mf, SEXP opt_reg);

RcppExport SEXP fqtl_rcpp_train_mf_cis_aux(SEXP y, SEXP x_m, SEXP a_m, SEXP c_m,
                                           SEXP x_v, SEXP opt_mf, SEXP opt_reg);

// Regeression routines

RcppExport SEXP fqtl_rcpp_train_reg(SEXP y, SEXP x_m, SEXP c_m, SEXP x_v,
                                    SEXP opt);

RcppExport SEXP fqtl_rcpp_train_reg_cis(SEXP y, SEXP x_m, SEXP a_x_m, SEXP c_m,
                                        SEXP x_v, SEXP opt);

RcppExport SEXP fqtl_rcpp_train_reg_cis_cis(SEXP y, SEXP x_m, SEXP a_x_m,
                                            SEXP c_m, SEXP a_c_m, SEXP x_v,
                                            SEXP opt);

// Factored regeression routines

RcppExport SEXP fqtl_rcpp_train_freg(SEXP y, SEXP x_m, SEXP c_m, SEXP x_v,
                                     SEXP opt);

RcppExport SEXP fqtl_rcpp_train_freg_cis(SEXP y, SEXP x_m, SEXP c_m, SEXP a_c_m,
                                         SEXP x_v, SEXP opt);

RcppExport SEXP fqtl_adj(SEXP d1, SEXP d2_start, SEXP d2_end, SEXP cis);

////////////////////////////
// Actual implementations //
////////////////////////////

using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor>;
using Scalar = Mat::Scalar;
using Index = Mat::Index;

// Factorization routines
template <typename ModelTag>
Rcpp::List rcpp_train_mf(const Mat &yy, const Mat &xx_mean, const Mat &xx_var,
                         const Rcpp::List &option_mf_list,
                         const Rcpp::List &option_reg_list);

template <typename ModelTag>
Rcpp::List rcpp_train_mf_cis(const Mat &yy, const Mat &xx_mean,
                             const SpMat &adj_mean, const Mat &xx_var,
                             const Rcpp::List &option_mf_list,
                             const Rcpp::List &option_reg_list);

template <typename ModelTag>
Rcpp::List rcpp_train_mf_cis_aux(const Mat &yy, const Mat &xx_sparse_mean,
                                 const SpMat &adj_mean,
                                 const Mat &xx_dense_mean, const Mat &xx_var,
                                 const Rcpp::List &option_mf_list,
                                 const Rcpp::List &option_reg_list);

// Regeression routines

template <typename ModelTag>
Rcpp::List rcpp_train_regression(const Mat &yy, const Mat &xx_mean,
                                 const Mat &cc_mean, const Mat &xx_var,
                                 const Rcpp::List &option_list);

template <typename ModelTag>
Rcpp::List rcpp_train_regression_cis(const Mat &yy, const Mat &xx_mean,
                                     const SpMat &adj_mean, const Mat &cc_mean,
                                     const Mat &xx_var,
                                     const Rcpp::List &option_list);

template <typename ModelTag>
Rcpp::List rcpp_train_regression_cis_cis(const Mat &yy, const Mat &xx_mean,
                                         const SpMat &adj_xx_mean,
                                         const Mat &cc_mean,
                                         const SpMat &adj_cc_mean,
                                         const Mat &xx_var,
                                         const Rcpp::List &option_list);

// Factored regeression routines

template <typename ModelTag>
Rcpp::List rcpp_train_factored_regression(const Mat &yy, const Mat &xx_mean,
                                          const Mat &cc_mean, const Mat &xx_var,
                                          const Rcpp::List &option_list);

template <typename ModelTag>
Rcpp::List rcpp_train_factored_regression_cis(const Mat &yy, const Mat &xx_mean,
                                              const Mat &cc_mean,
                                              const SpMat &adj_cc_mean,
                                              const Mat &xx_var,
                                              const Rcpp::List &option_list);

////////////////////////////////////////////////////////////////

#define ASSERT_LIST_RET(cond, msg) \
  if (!(cond)) {                   \
    ELOG(msg);                     \
    return Rcpp::List::create();   \
  }

void set_options_from_list(const Rcpp::List &_list, options_t &opt);

template <typename Derived, typename OtherDerived>
void summarize_llik(const Eigen::MatrixBase<Derived> &llik,
                    Eigen::MatrixBase<OtherDerived> &llik_trace);

Rcpp::List rcpp_fqtl_adj_list(const Rcpp::NumericVector &d1_loc,
                              const Rcpp::NumericVector &d2_start_loc,
                              const Rcpp::NumericVector &d2_end_loc,
                              const double cis_window);

template <typename T>
Rcpp::List param_rcpp_list(const T &param);

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_spike_slab);

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_col_spike_slab);

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_col_slab);

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_slab);

////////////////////////////////////////////////////////////////

struct m_gaussian_tag {};
struct m_voom_tag {};
struct m_nb_tag {};

template <typename Mat, typename ModelT>
struct impl_model_maker_t;

template <typename Mat>
struct impl_model_maker_t<Mat, m_gaussian_tag> {
  using Scalar = typename Mat::Scalar;
  using model_type = gaussian_model_t<Mat>;

  std::shared_ptr<model_type> operator()(const Mat &Y) {
    calc_stat_t<Scalar> stat_op;
    visit(Y, stat_op);
    const Scalar vmax = stat_op.var();
    const Scalar vmin = 1e-4 * vmax;
    TLOG("gaussian model : Vmax " << vmax << ", Vmin " << vmin);
    return std::make_shared<model_type>(Y, typename model_type::Vmin_t(vmin),
                                        typename model_type::Vmax_t(vmax));
  }
};

template <typename Mat>
struct impl_model_maker_t<Mat, m_nb_tag> {
  using model_type = nb_model_t<Mat>;
  std::shared_ptr<model_type> operator()(const Mat &Y) {
    TLOG("negative binomial model");
    return std::make_shared<model_type>(Y);
  }
};

template <typename Mat>
struct impl_model_maker_t<Mat, m_voom_tag> {
  using Scalar = typename Mat::Scalar;
  using model_type = gaussian_voom_model_t<Mat>;

  std::shared_ptr<model_type> operator()(const Mat &Y) {
    calc_stat_t<Scalar> stat_op;
    visit(Y, stat_op);
    const Scalar vmax = stat_op.var();
    const Scalar vmin = 1e-4 * vmax;
    TLOG("voom model : Vmax " << vmax << ", Vmin " << vmin);
    return std::make_shared<model_type>(Y, typename model_type::Vmin_t(vmin),
                                        typename model_type::Vmax_t(vmax));
  }
};

template <typename T, typename Mat>
auto make_model(const Mat &Y) {
  impl_model_maker_t<Mat, T> maker;
  return maker(Y);
}

const std::string get_model_name(const Rcpp::List &_list);

#endif
