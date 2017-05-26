#include "rcpp_fqtl.hh"

#define ASSERT_LIST_RET(cond, msg)                                             \
  if (!(cond)) {                                                               \
    ELOG(msg);                                                                 \
    return Rcpp::List::create();                                               \
  }

////////////////////////////////////////////////////////////////
// mean ~ U * V' + xx_mean * theta
// var ~ xx_var * theta
// [[Rcpp::export(name="fqtl.mf", rng=false)]]
Rcpp::List
rcpp_train_mf(const Mat &yy,      // n x m
              const Mat &xx_mean, // n x p -> regression -> [n x p] [p x m]
              const Mat &xx_var,  // n x q -> regression -> [n x q] [q x m]
              const Rcpp::List &option_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");

  options_t opt;
  set_options_from_list(option_list, opt);

  const Index n = yy.rows();
  const Index m = yy.cols();
  const Index K = opt.k();

  auto mf_theta_u = make_dense_col_spike_slab<Scalar>(n, K, opt);
  auto mf_theta_v = make_dense_col_spike_slab<Scalar>(m, K, opt);
  auto mf_eta = make_factorization_eta(yy, mf_theta_u, mf_theta_v);

  if (opt.mf_svd_init()) {
    mf_eta.init_by_svd(yy, opt.jitter());
  } else {
    std::mt19937 rng(opt.rseed());
    mf_eta.jitter(opt.jitter(), rng);
  }

  auto mean_theta = make_dense_spike_slab<Scalar>(xx_mean.cols(), m, opt);
  auto mean_eta = make_regression_eta(xx_mean, yy, mean_theta);

  auto var_theta = make_dense_col_slab<Scalar>(xx_var.cols(), m, opt);
  auto var_eta = make_regression_eta(xx_var, yy, var_theta);

  auto model_ptr = make_model<m_gaussian_tag>(yy);
  auto &model = *model_ptr.get();
  auto llik_trace = impl_fit_eta(model, opt, std::make_tuple(mf_eta, mean_eta),
                                 std::make_tuple(var_eta));

  TLOG("Finished MF");

  return Rcpp::List::create(Rcpp::_["U"] = param_rcpp_list(mf_theta_u),
                            Rcpp::_["V"] = param_rcpp_list(mf_theta_v),
                            Rcpp::_["mean"] = param_rcpp_list(mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(var_theta),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ U * V' + xx_mean * theta
// var ~ xx_var * theta
// [[Rcpp::export(name="fqtl.mf.cis", rng=false)]]
Rcpp::List
rcpp_train_mf_cis(const Mat &yy,      // n x m
                  const Mat &xx_mean, // n x p -> regression -> [n x p] [p x m]
                  const SpMat &adj_mean, // p x m adjacency
                  const Mat &xx_var, // n x q -> regression -> [n x q] [q x m]
                  const Rcpp::List &option_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");
  ASSERT_LIST_RET(xx_mean.cols() == adj_mean.rows(),
                  "xx_mean and adj_mean with different number of variables");
  ASSERT_LIST_RET(yy.cols() == adj_mean.cols(),
                  "yy and adj_mean with different number of outputs");

  options_t opt;
  set_options_from_list(option_list, opt);

  const Index n = yy.rows();
  const Index m = yy.cols();
  const Index K = opt.k();

  auto mf_theta_u = make_dense_col_spike_slab<Scalar>(n, K, opt);
  auto mf_theta_v = make_dense_col_spike_slab<Scalar>(m, K, opt);
  auto mf_eta = make_factorization_eta(yy, mf_theta_u, mf_theta_v);
  if (opt.mf_svd_init()) {
    mf_eta.init_by_svd(yy, opt.jitter());
  } else {
    std::mt19937 rng(opt.rseed());
    mf_eta.jitter(opt.jitter(), rng);
  }

  auto mean_theta = make_sparse_spike_slab<Scalar>(adj_mean, opt);
  auto mean_eta = make_regression_eta(xx_mean, yy, mean_theta);

  auto var_theta = make_dense_slab<Scalar>(xx_var.cols(), m, opt);
  auto var_eta = make_regression_eta(xx_var, yy, var_theta);

  auto model_ptr = make_model<m_gaussian_tag>(yy);
  auto &model = *model_ptr.get();
  auto llik_trace = impl_fit_eta(model, opt, std::make_tuple(mf_eta, mean_eta),
                                 std::make_tuple(var_eta));

  TLOG("Finished MF");

  return Rcpp::List::create(Rcpp::_["U"] = param_rcpp_list(mf_theta_u),
                            Rcpp::_["V"] = param_rcpp_list(mf_theta_v),
                            Rcpp::_["mean"] = param_rcpp_list(mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(var_theta),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ U * V' + xx_sparse_mean * theta + xx_dense_mean * theta
// var ~ xx_var * theta
// [[Rcpp::export(name="fqtl.mf.cis.aux", rng=false)]]
Rcpp::List rcpp_train_mf_cis_aux(
    const Mat &yy,             // n x m
    const Mat &xx_sparse_mean, // n x p
    const SpMat &adj_mean,     // p x m adjacency
    const Mat &xx_dense_mean,  // additional covariates
    const Mat &xx_var,         // n x q -> regression -> [n x q] [q x m]
    const Rcpp::List &option_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_sparse_mean.rows(),
                  "yy and xx_sparse_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");
  ASSERT_LIST_RET(
      xx_sparse_mean.cols() == adj_mean.rows(),
      "xx_sparse_mean and adj_mean with different number of variables");
  ASSERT_LIST_RET(yy.cols() == adj_mean.cols(),
                  "yy and adj_mean with different number of outputs");

  options_t opt;
  set_options_from_list(option_list, opt);

  const Index n = yy.rows();
  const Index m = yy.cols();
  const Index K = opt.k();

  auto mf_theta_u = make_dense_col_spike_slab<Scalar>(n, K, opt);
  auto mf_theta_v = make_dense_col_spike_slab<Scalar>(m, K, opt);
  auto mf_eta = make_factorization_eta(yy, mf_theta_u, mf_theta_v);
  if (opt.mf_svd_init()) {
    mf_eta.init_by_svd(yy, opt.jitter());
  } else {
    std::mt19937 rng(opt.rseed());
    mf_eta.jitter(opt.jitter(), rng);
  }

  auto mean_sparse_theta = make_sparse_spike_slab<Scalar>(adj_mean, opt);
  auto mean_sparse_eta =
      make_regression_eta(xx_sparse_mean, yy, mean_sparse_theta);

  auto mean_dense_theta =
      make_dense_spike_slab<Scalar>(xx_dense_mean.cols(), m, opt);
  auto mean_dense_eta =
      make_regression_eta(xx_dense_mean, yy, mean_dense_theta);

  auto var_theta = make_dense_slab<Scalar>(xx_var.cols(), m, opt);
  auto var_eta = make_regression_eta(xx_var, yy, var_theta);

  auto model_ptr = make_model<m_gaussian_tag>(yy);
  auto &model = *model_ptr.get();
  auto llik_trace = impl_fit_eta(
      model, opt, std::make_tuple(mf_eta, mean_sparse_eta, mean_dense_eta),
      std::make_tuple(var_eta));

  TLOG("Finished MF");

  return Rcpp::List::create(
      Rcpp::_["U"] = param_rcpp_list(mf_theta_u),
      Rcpp::_["V"] = param_rcpp_list(mf_theta_v),
      Rcpp::_["mean.sparse"] = param_rcpp_list(mean_sparse_theta),
      Rcpp::_["mean.dense"] = param_rcpp_list(mean_dense_theta),
      Rcpp::_["var"] = param_rcpp_list(var_theta),
      Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * U * V' + C * theta + theta * Ct
// var  ~ Xv * theta
// [[Rcpp::export(name="fqtl.regression.factored", rng=false)]]
Rcpp::List rcpp_train_factored_regression(const Mat &yy,      // n x m
                                          const Mat &xx_mean, // n x p -> p x m
                                          const Mat &cc_mean, // n x p -> p x m
                                          const Mat &xx_var,  // n x p
                                          const Rcpp::List &option_list) {
  ///////////////////
  // check options //
  ///////////////////

  if (yy.cols() < 2)
    WLOG("Factored regression with 1 output");

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == cc_mean.rows(),
                  "yy and cc_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");

  options_t opt;
  set_options_from_list(option_list, opt);

  //////////////////////////
  // construct parameters //
  //////////////////////////

  const Index p = xx_mean.cols();
  const Index m = yy.cols();
  const Index K = opt.k();

  auto c_mean_theta =
      make_dense_spike_slab<Scalar>(cc_mean.cols(), yy.cols(), opt);
  auto x_var_theta = make_dense_slab<Scalar>(xx_var.cols(), yy.cols(), opt);

  auto c_mean_eta = make_regression_eta(cc_mean, yy, c_mean_theta);
  auto x_var_eta = make_regression_eta(xx_var, yy, x_var_theta);

  auto mf_theta_u = make_dense_spike_slab<Scalar>(p, K, opt);
  auto mf_theta_v = make_dense_spike_slab<Scalar>(m, K, opt);
  auto mean_eta =
      make_factored_regression_eta(xx_mean, yy, mf_theta_u, mf_theta_v);

  std::mt19937 rng(opt.rseed());
  mean_eta.jitter(opt.jitter(), rng);

  auto model_ptr = make_model<m_gaussian_tag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  return Rcpp::List::create(Rcpp::_["mean.left"] = param_rcpp_list(mf_theta_u),
                            Rcpp::_["mean.right"] = param_rcpp_list(mf_theta_v),
                            Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(x_var_theta),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * U * V' + C * theta
// var  ~ Xv * theta
// [[Rcpp::export(name="fqtl.regression.factored.cis", rng=false)]]
Rcpp::List
rcpp_train_factored_regression_cis(const Mat &yy,            // n x m
                                   const Mat &xx_mean,       // n x p -> p x m
                                   const Mat &cc_mean,       // n x p -> p x m
                                   const SpMat &adj_cc_mean, // p x m
                                   const Mat &xx_var,        // n x p
                                   const Rcpp::List &option_list) {
  ///////////////////
  // check options //
  ///////////////////

  if (yy.cols() < 2)
    WLOG("Factored regression with 1 output");

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == cc_mean.rows(),
                  "yy and cc_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");
  ASSERT_LIST_RET(yy.cols() == adj_cc_mean.cols(),
                  "yy and adj_cc_mean with different number of outputs");
  ASSERT_LIST_RET(cc_mean.cols() == adj_cc_mean.rows(),
                  "cc_mean and adj_cc_mean with different number of vars");

  options_t opt;
  set_options_from_list(option_list, opt);

  //////////////////////////
  // construct parameters //
  //////////////////////////

  const Index p = xx_mean.cols();
  const Index m = yy.cols();
  const Index K = opt.k();

  auto c_mean_theta = make_sparse_spike_slab<Scalar>(adj_cc_mean, opt);
  auto x_var_theta = make_dense_slab<Scalar>(xx_var.cols(), yy.cols(), opt);

  auto c_mean_eta = make_regression_eta(cc_mean, yy, c_mean_theta);
  auto x_var_eta = make_regression_eta(xx_var, yy, x_var_theta);

  auto mf_theta_u = make_dense_spike_slab<Scalar>(p, K, opt);
  auto mf_theta_v = make_dense_spike_slab<Scalar>(m, K, opt);
  auto mean_eta =
      make_factored_regression_eta(xx_mean, yy, mf_theta_u, mf_theta_v);

  std::mt19937 rng(opt.rseed());
  mean_eta.jitter(opt.jitter(), rng);

  auto model_ptr = make_model<m_gaussian_tag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  return Rcpp::List::create(Rcpp::_["mean.left"] = param_rcpp_list(mf_theta_u),
                            Rcpp::_["mean.right"] = param_rcpp_list(mf_theta_v),
                            Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(x_var_theta),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * theta + C * theta
// var  ~ Xv * theta
// [[Rcpp::export(name="fqtl.regression", rng=false)]]
Rcpp::List rcpp_train_regression(const Mat &yy,      // n x m
                                 const Mat &xx_mean, // n x p
                                 const Mat &cc_mean, // n x p
                                 const Mat &xx_var,  // n x p
                                 const Rcpp::List &option_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == cc_mean.rows(),
                  "yy and cc_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");

  options_t opt;
  set_options_from_list(option_list, opt);

  auto c_mean_theta =
      make_dense_spike_slab<Scalar>(cc_mean.cols(), yy.cols(), opt);
  auto x_var_theta = make_dense_slab<Scalar>(xx_var.cols(), yy.cols(), opt);

  auto c_mean_eta = make_regression_eta(cc_mean, yy, c_mean_theta);
  auto x_var_eta = make_regression_eta(xx_var, yy, x_var_theta);

  auto mean_theta =
      make_dense_spike_slab<Scalar>(xx_mean.cols(), yy.cols(), opt);
  auto mean_eta = make_regression_eta(xx_mean, yy, mean_theta);

  auto model_ptr = make_model<m_gaussian_tag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  // residual calculation
  auto theta_resid_full =
      make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  auto theta_resid_cov = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  if (opt.out_resid()) {
    {
      auto resid_eta = make_residual_eta(yy, theta_resid_full);
      impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                   std::make_tuple(x_var_eta),
                   std::make_tuple(mean_eta, c_mean_eta));
    }
    {
      auto resid_eta = make_residual_eta(yy, theta_resid_cov);
      impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                   std::make_tuple(x_var_eta), std::make_tuple(mean_eta));
    }
  }

  return Rcpp::List::create(
      Rcpp::_["mean"] = param_rcpp_list(mean_theta),
      Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
      Rcpp::_["var"] = param_rcpp_list(x_var_theta),
      Rcpp::_["resid.full"] = param_rcpp_list(theta_resid_full),
      Rcpp::_["resid.no.mean"] = param_rcpp_list(theta_resid_cov),
      Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * theta + C * theta
// var  ~ Xv * theta
// [[Rcpp::export(name="fqtl.regression.cis", rng=false)]]
Rcpp::List rcpp_train_regression_cis(const Mat &yy,         // n x m
                                     const Mat &xx_mean,    // n x p
                                     const SpMat &adj_mean, // p x m
                                     const Mat &cc_mean,    // n x p
                                     const Mat &xx_var,     // n x p
                                     const Rcpp::List &option_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == cc_mean.rows(),
                  "yy and cc_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");
  ASSERT_LIST_RET(xx_mean.cols() == adj_mean.rows(),
                  "xx_mean and adj_mean with different number of variables");
  ASSERT_LIST_RET(yy.cols() == adj_mean.cols(),
                  "yy and adj_mean with different number of outputs");

  options_t opt;
  set_options_from_list(option_list, opt);

  auto mean_theta = make_sparse_spike_slab<Scalar>(adj_mean, opt);
  auto c_mean_theta =
      make_dense_spike_slab<Scalar>(cc_mean.cols(), yy.cols(), opt);
  auto x_var_theta = make_dense_slab<Scalar>(xx_var.cols(), yy.cols(), opt);

  auto mean_eta = make_regression_eta(xx_mean, yy, mean_theta);
  auto c_mean_eta = make_regression_eta(cc_mean, yy, c_mean_theta);
  auto x_var_eta = make_regression_eta(xx_var, yy, x_var_theta);

  auto model_ptr = make_model<m_gaussian_tag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  // residual calculation
  auto theta_resid_full =
      make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  auto theta_resid_cov = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  if (opt.out_resid()) {
    {
      auto resid_eta = make_residual_eta(yy, theta_resid_full);
      impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                   std::make_tuple(x_var_eta),
                   std::make_tuple(mean_eta, c_mean_eta));
    }
    {
      auto resid_eta = make_residual_eta(yy, theta_resid_cov);
      impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                   std::make_tuple(x_var_eta), std::make_tuple(mean_eta));
    }
  }

  return Rcpp::List::create(
      Rcpp::_["mean"] = param_rcpp_list(mean_theta),
      Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
      Rcpp::_["var"] = param_rcpp_list(x_var_theta),
      Rcpp::_["resid.full"] = param_rcpp_list(theta_resid_full),
      Rcpp::_["resid.no.mean"] = param_rcpp_list(theta_resid_cov),
      Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * theta + C * theta
// var  ~ Xv * theta
// [[Rcpp::export(name="fqtl.regression.cis.cis", rng=false)]]
Rcpp::List rcpp_train_regression_cis_cis(const Mat &yy,            // n x m
                                         const Mat &xx_mean,       // n x p
                                         const SpMat &adj_xx_mean, // p x m
                                         const Mat &cc_mean,       // n x p
                                         const SpMat &adj_cc_mean, // p x m
                                         const Mat &xx_var,        // n x p
                                         const Rcpp::List &option_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == cc_mean.rows(),
                  "yy and cc_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");
  ASSERT_LIST_RET(xx_mean.cols() == adj_xx_mean.rows(),
                  "xx_mean and adj_xx_mean with different number of variables");
  ASSERT_LIST_RET(yy.cols() == adj_xx_mean.cols(),
                  "yy and adj_xx_mean with different number of outputs");
  ASSERT_LIST_RET(cc_mean.cols() == adj_cc_mean.rows(),
                  "cc_mean and adj_cc_mean with different number of variables");
  ASSERT_LIST_RET(yy.cols() == adj_cc_mean.cols(),
                  "yy and adj_cc_mean with different number of outputs");

  options_t opt;
  set_options_from_list(option_list, opt);

  auto mean_theta = make_sparse_spike_slab<Scalar>(adj_xx_mean, opt);
  auto c_mean_theta = make_sparse_spike_slab<Scalar>(adj_cc_mean, opt);
  auto x_var_theta = make_dense_slab<Scalar>(xx_var.cols(), yy.cols(), opt);

  auto mean_eta = make_regression_eta(xx_mean, yy, mean_theta);
  auto c_mean_eta = make_regression_eta(cc_mean, yy, c_mean_theta);
  auto x_var_eta = make_regression_eta(xx_var, yy, x_var_theta);

  auto model_ptr = make_model<m_gaussian_tag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  // residual calculation
  auto theta_resid_full =
      make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  auto theta_resid_cov = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  if (opt.out_resid()) {
    {
      auto resid_eta = make_residual_eta(yy, theta_resid_full);
      impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                   std::make_tuple(x_var_eta),
                   std::make_tuple(mean_eta, c_mean_eta));
    }
    {
      auto resid_eta = make_residual_eta(yy, theta_resid_cov);
      impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                   std::make_tuple(x_var_eta), std::make_tuple(mean_eta));
    }
  }

  return Rcpp::List::create(
      Rcpp::_["mean"] = param_rcpp_list(mean_theta),
      Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
      Rcpp::_["var"] = param_rcpp_list(x_var_theta),
      Rcpp::_["resid.full"] = param_rcpp_list(theta_resid_full),
      Rcpp::_["resid.no.mean"] = param_rcpp_list(theta_resid_cov),
      Rcpp::_["llik"] = llik_trace);
}

using namespace Rcpp;

////////////////////
// Package export //
////////////////////

RcppExport SEXP fqtl_rcpp_train_mf(SEXP y, SEXP x_m, SEXP x_v, SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat &>::type yy(y);
  Rcpp::traits::input_parameter<const Mat &>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat &>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  __result = Rcpp::wrap(rcpp_train_mf(yy, xx_m, xx_v, option_list));

  return __result;
  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_mf_cis(SEXP y, SEXP x_m, SEXP a_m, SEXP x_v,
                                       SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat &>::type yy(y);
  Rcpp::traits::input_parameter<const Mat &>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const SpMat &>::type aa_m(a_m);
  Rcpp::traits::input_parameter<const Mat &>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  __result = Rcpp::wrap(rcpp_train_mf_cis(yy, xx_m, aa_m, xx_v, option_list));

  return __result;
  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_mf_cis_aux(SEXP y, SEXP x_m, SEXP a_m, SEXP c_m,
                                           SEXP x_v, SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat &>::type yy(y);
  Rcpp::traits::input_parameter<const Mat &>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const SpMat &>::type aa_m(a_m);
  Rcpp::traits::input_parameter<const Mat &>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat &>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  __result = Rcpp::wrap(
      rcpp_train_mf_cis_aux(yy, xx_m, aa_m, cc_m, xx_v, option_list));

  return __result;
  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_reg(SEXP y, SEXP x_m, SEXP c_m, SEXP x_v,
                                    SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat &>::type yy(y);
  Rcpp::traits::input_parameter<const Mat &>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat &>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat &>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  __result =
      Rcpp::wrap(rcpp_train_regression(yy, xx_m, cc_m, xx_v, option_list));

  return __result;
  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_reg_cis(SEXP y, SEXP x_m, SEXP a_x_m, SEXP c_m,
                                        SEXP x_v, SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat &>::type yy(y);
  Rcpp::traits::input_parameter<const Mat &>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const SpMat &>::type adj_xx_m(a_x_m);
  Rcpp::traits::input_parameter<const Mat &>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat &>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  __result = Rcpp::wrap(
      rcpp_train_regression_cis(yy, xx_m, adj_xx_m, cc_m, xx_v, option_list));

  return __result;
  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_reg_cis_cis(SEXP y, SEXP x_m, SEXP a_x_m,
                                            SEXP c_m, SEXP a_c_m, SEXP x_v,
                                            SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat &>::type yy(y);
  Rcpp::traits::input_parameter<const Mat &>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const SpMat &>::type adj_xx_m(a_x_m);
  Rcpp::traits::input_parameter<const Mat &>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const SpMat &>::type adj_cc_m(a_c_m);
  Rcpp::traits::input_parameter<const Mat &>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  __result = Rcpp::wrap(rcpp_train_regression_cis_cis(
      yy, xx_m, adj_xx_m, cc_m, adj_cc_m, xx_v, option_list));

  return __result;
  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_freg(SEXP y, SEXP x_m, SEXP c_m, SEXP x_v,
                                     SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat &>::type yy(y);
  Rcpp::traits::input_parameter<const Mat &>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat &>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat &>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  __result = Rcpp::wrap(
      rcpp_train_factored_regression(yy, xx_m, cc_m, xx_v, option_list));

  return __result;
  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_freg_cis(SEXP y, SEXP x_m, SEXP c_m, SEXP a_c_m,
                                         SEXP x_v, SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat &>::type yy(y);
  Rcpp::traits::input_parameter<const Mat &>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat &>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const SpMat &>::type adj_cc_m(a_c_m);
  Rcpp::traits::input_parameter<const Mat &>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  __result = Rcpp::wrap(rcpp_train_factored_regression_cis(
      yy, xx_m, cc_m, adj_cc_m, xx_v, option_list));

  return __result;
  END_RCPP
}

RcppExport SEXP fqtl_adj(SEXP d1, SEXP d2_start, SEXP d2_end, SEXP cis) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Rcpp::NumericVector &>::type d1_loc(d1);
  Rcpp::traits::input_parameter<const Rcpp::NumericVector &>::type d2_start_loc(
      d2_start);
  Rcpp::traits::input_parameter<const Rcpp::NumericVector &>::type d2_end_loc(
      d2_end);
  Rcpp::traits::input_parameter<const double>::type cis_window(cis);

  __result = Rcpp::wrap(
      rcpp_fqtl_adj_list(d1_loc, d2_start_loc, d2_end_loc, cis_window));

  return __result;
  END_RCPP
}
