#include "rcpp_fqtl.hh"

using namespace Rcpp;

////////////////////
// Package export //
////////////////////

RcppExport SEXP fqtl_rcpp_train_mf(SEXP y, SEXP x_m, SEXP x_v, SEXP opt_mf,
                                   SEXP opt_reg) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::List option_mf_list(opt_mf);
  Rcpp::List option_reg_list(opt_reg);

  const auto model = get_model_name(option_mf_list);

  if (model == "nb") {
    return Rcpp::wrap(rcpp_train_mf<m_nb_tag>(yy, xx_m, xx_v, option_mf_list,
                                              option_reg_list));
  } else if (model == "logit") {
    return Rcpp::wrap(rcpp_train_mf<m_logit_tag>(yy, xx_m, xx_v, option_mf_list,
                                                 option_reg_list));
  } else if (model == "voom") {
    return Rcpp::wrap(rcpp_train_mf<m_voom_tag>(yy, xx_m, xx_v, option_mf_list,
                                                option_reg_list));

  } else if (model == "beta") {
    return Rcpp::wrap(rcpp_train_mf<m_beta_tag>(yy, xx_m, xx_v, option_mf_list,
                                                option_reg_list));

  } else {
    return Rcpp::wrap(rcpp_train_mf<m_gaussian_tag>(
        yy, xx_m, xx_v, option_mf_list, option_reg_list));
  }

  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_mf_cis(SEXP y, SEXP x_m, SEXP a_m, SEXP x_v,
                                       SEXP opt_mf, SEXP opt_reg) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const SpMat>::type aa_m(a_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::List option_mf_list(opt_mf);
  Rcpp::List option_reg_list(opt_reg);

  const auto model = get_model_name(option_mf_list);

  if (model == "nb") {
    return Rcpp::wrap(rcpp_train_mf_cis<m_nb_tag>(
        yy, xx_m, aa_m, xx_v, option_mf_list, option_reg_list));
  } else if (model == "logit") {
    return Rcpp::wrap(rcpp_train_mf_cis<m_logit_tag>(
        yy, xx_m, aa_m, xx_v, option_mf_list, option_reg_list));
  } else if (model == "voom") {
    return Rcpp::wrap(rcpp_train_mf_cis<m_voom_tag>(
        yy, xx_m, aa_m, xx_v, option_mf_list, option_reg_list));

  } else if (model == "beta") {
    return Rcpp::wrap(rcpp_train_mf_cis<m_beta_tag>(
        yy, xx_m, aa_m, xx_v, option_mf_list, option_reg_list));

  } else {
    return Rcpp::wrap(rcpp_train_mf_cis<m_gaussian_tag>(
        yy, xx_m, aa_m, xx_v, option_mf_list, option_reg_list));
  }

  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_mf_cis_aux(SEXP y, SEXP x_m, SEXP a_m, SEXP c_m,
                                           SEXP x_v, SEXP opt_mf,
                                           SEXP opt_reg) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const SpMat>::type aa_m(a_m);
  Rcpp::traits::input_parameter<const Mat>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::List option_mf_list(opt_mf);
  Rcpp::List option_reg_list(opt_reg);

  const auto model = get_model_name(option_mf_list);

  if (model == "nb") {
    return Rcpp::wrap(rcpp_train_mf_cis_aux<m_nb_tag>(
        yy, xx_m, aa_m, cc_m, xx_v, option_mf_list, option_reg_list));

  } else if (model == "logit") {
    return Rcpp::wrap(rcpp_train_mf_cis_aux<m_logit_tag>(
        yy, xx_m, aa_m, cc_m, xx_v, option_mf_list, option_reg_list));

  } else if (model == "voom") {
    return Rcpp::wrap(rcpp_train_mf_cis_aux<m_voom_tag>(
        yy, xx_m, aa_m, cc_m, xx_v, option_mf_list, option_reg_list));

  } else if (model == "beta") {
    return Rcpp::wrap(rcpp_train_mf_cis_aux<m_beta_tag>(
        yy, xx_m, aa_m, cc_m, xx_v, option_mf_list, option_reg_list));

  } else {
    return Rcpp::wrap(rcpp_train_mf_cis_aux<m_gaussian_tag>(
        yy, xx_m, aa_m, cc_m, xx_v, option_mf_list, option_reg_list));
  }

  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_reg(SEXP y, SEXP x_m, SEXP c_m, SEXP x_v,
                                    SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  const auto model = get_model_name(option_list);

  if (model == "nb") {
    return Rcpp::wrap(
        rcpp_train_regression<m_nb_tag>(yy, xx_m, cc_m, xx_v, option_list));

  } else if (model == "logit") {
    return Rcpp::wrap(
        rcpp_train_regression<m_logit_tag>(yy, xx_m, cc_m, xx_v, option_list));

  } else if (model == "voom") {
    return Rcpp::wrap(
        rcpp_train_regression<m_voom_tag>(yy, xx_m, cc_m, xx_v, option_list));

  } else if (model == "beta") {
    return Rcpp::wrap(
        rcpp_train_regression<m_beta_tag>(yy, xx_m, cc_m, xx_v, option_list));

  } else {
    return Rcpp::wrap(rcpp_train_regression<m_gaussian_tag>(yy, xx_m, cc_m,
                                                            xx_v, option_list));
  }

  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_reg_cis(SEXP y, SEXP x_m, SEXP a_x_m, SEXP c_m,
                                        SEXP x_v, SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const SpMat>::type adj_xx_m(a_x_m);
  Rcpp::traits::input_parameter<const Mat>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  const auto model = get_model_name(option_list);

  if (model == "nb") {
    return Rcpp::wrap(rcpp_train_regression_cis<m_nb_tag>(
        yy, xx_m, adj_xx_m, cc_m, xx_v, option_list));

  } else if (model == "logit") {
    return Rcpp::wrap(rcpp_train_regression_cis<m_logit_tag>(
        yy, xx_m, adj_xx_m, cc_m, xx_v, option_list));

  } else if (model == "voom") {
    return Rcpp::wrap(rcpp_train_regression_cis<m_voom_tag>(
        yy, xx_m, adj_xx_m, cc_m, xx_v, option_list));

  } else if (model == "beta") {
    return Rcpp::wrap(rcpp_train_regression_cis<m_beta_tag>(
        yy, xx_m, adj_xx_m, cc_m, xx_v, option_list));

  } else {
    return Rcpp::wrap(rcpp_train_regression_cis<m_gaussian_tag>(
        yy, xx_m, adj_xx_m, cc_m, xx_v, option_list));
  }

  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_reg_cis_cis(SEXP y, SEXP x_m, SEXP a_x_m,
                                            SEXP c_m, SEXP a_c_m, SEXP x_v,
                                            SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const SpMat>::type adj_xx_m(a_x_m);
  Rcpp::traits::input_parameter<const Mat>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const SpMat>::type adj_cc_m(a_c_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  const auto model = get_model_name(option_list);

  if (model == "nb") {
    return Rcpp::wrap(rcpp_train_regression_cis_cis<m_nb_tag>(
        yy, xx_m, adj_xx_m, cc_m, adj_cc_m, xx_v, option_list));
  } else if (model == "logit") {
    return Rcpp::wrap(rcpp_train_regression_cis_cis<m_logit_tag>(
        yy, xx_m, adj_xx_m, cc_m, adj_cc_m, xx_v, option_list));

  } else if (model == "voom") {
    return Rcpp::wrap(rcpp_train_regression_cis_cis<m_voom_tag>(
        yy, xx_m, adj_xx_m, cc_m, adj_cc_m, xx_v, option_list));

  } else if (model == "beta") {
    return Rcpp::wrap(rcpp_train_regression_cis_cis<m_beta_tag>(
        yy, xx_m, adj_xx_m, cc_m, adj_cc_m, xx_v, option_list));

  } else {
    return Rcpp::wrap(rcpp_train_regression_cis_cis<m_gaussian_tag>(
        yy, xx_m, adj_xx_m, cc_m, adj_cc_m, xx_v, option_list));
  }

  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_freg(SEXP y, SEXP x_m, SEXP c_m, SEXP x_v,
                                     SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  const auto model = get_model_name(option_list);

  if (model == "nb") {
    return Rcpp::wrap(rcpp_train_factored_regression<m_nb_tag>(
        yy, xx_m, cc_m, xx_v, option_list));

  } else if (model == "logit") {
    return Rcpp::wrap(rcpp_train_factored_regression<m_logit_tag>(
        yy, xx_m, cc_m, xx_v, option_list));

  } else if (model == "voom") {
    return Rcpp::wrap(rcpp_train_factored_regression<m_voom_tag>(
        yy, xx_m, cc_m, xx_v, option_list));

  } else if (model == "beta") {
    return Rcpp::wrap(rcpp_train_factored_regression<m_beta_tag>(
        yy, xx_m, cc_m, xx_v, option_list));

  } else {
    return Rcpp::wrap(rcpp_train_factored_regression<m_gaussian_tag>(
        yy, xx_m, cc_m, xx_v, option_list));
  }

  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_fwreg(SEXP y, SEXP x_m, SEXP c_m, SEXP x_v,
                                      SEXP w, SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::traits::input_parameter<const Mat>::type ww(w);
  Rcpp::List option_list(opt);

  const auto model = get_model_name(option_list);

  if (model == "nb") {
    return Rcpp::wrap(rcpp_train_factored_regression<m_nb_tag>(
        yy, xx_m, cc_m, xx_v, ww, option_list));

  } else if (model == "logit") {
    return Rcpp::wrap(rcpp_train_factored_regression<m_logit_tag>(
        yy, xx_m, cc_m, xx_v, ww, option_list));

  } else if (model == "voom") {
    return Rcpp::wrap(rcpp_train_factored_regression<m_voom_tag>(
        yy, xx_m, cc_m, xx_v, ww, option_list));

  } else if (model == "beta") {
    return Rcpp::wrap(rcpp_train_factored_regression<m_beta_tag>(
        yy, xx_m, cc_m, xx_v, ww, option_list));

  } else {
    return Rcpp::wrap(rcpp_train_factored_regression<m_gaussian_tag>(
        yy, xx_m, cc_m, xx_v, ww, option_list));
  }

  END_RCPP
}

RcppExport SEXP fqtl_rcpp_train_freg_cis(SEXP y, SEXP x_m, SEXP c_m, SEXP a_c_m,
                                         SEXP x_v, SEXP opt) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::traits::input_parameter<const Mat>::type yy(y);
  Rcpp::traits::input_parameter<const Mat>::type xx_m(x_m);
  Rcpp::traits::input_parameter<const Mat>::type cc_m(c_m);
  Rcpp::traits::input_parameter<const SpMat>::type adj_cc_m(a_c_m);
  Rcpp::traits::input_parameter<const Mat>::type xx_v(x_v);
  Rcpp::List option_list(opt);

  const auto model = get_model_name(option_list);

  if (model == "nb") {
    return Rcpp::wrap(rcpp_train_factored_regression_cis<m_nb_tag>(
        yy, xx_m, cc_m, adj_cc_m, xx_v, option_list));

  } else if (model == "logit") {
    return Rcpp::wrap(rcpp_train_factored_regression_cis<m_logit_tag>(
        yy, xx_m, cc_m, adj_cc_m, xx_v, option_list));

  } else if (model == "voom") {
    return Rcpp::wrap(rcpp_train_factored_regression_cis<m_voom_tag>(
        yy, xx_m, cc_m, adj_cc_m, xx_v, option_list));

  } else if (model == "beta") {
    return Rcpp::wrap(rcpp_train_factored_regression_cis<m_beta_tag>(
        yy, xx_m, cc_m, adj_cc_m, xx_v, option_list));

  } else {
    return Rcpp::wrap(rcpp_train_factored_regression_cis<m_gaussian_tag>(
        yy, xx_m, cc_m, adj_cc_m, xx_v, option_list));
  }

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

  return Rcpp::wrap(
      rcpp_fqtl_adj_list(d1_loc, d2_start_loc, d2_end_loc, cis_window));

  END_RCPP
}

////////////////////////////
// Actual implementations //
////////////////////////////

const std::string get_model_name(const Rcpp::List &_list) {
  if (_list.containsElementNamed("model"))
    return Rcpp::as<std::string>(_list["model"]);
  return std::string("gaussian");
}

////////////////////////////////////////////////////////////////
// mean ~ U * V' + xx_mean * theta
// var ~ xx_var * theta
template <typename ModelTag>
Rcpp::List rcpp_train_mf(
    const Mat &yy,       // n x m
    const Mat &xx_mean,  // n x p -> regression -> [n x p] [p x m]
    const Mat &xx_var,   // n x q -> regression -> [n x q] [q x m]
    const Rcpp::List &option_mf_list, const Rcpp::List &option_reg_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");

  options_t opt_mf;
  options_t opt_reg;
  set_options_from_list(option_mf_list, opt_mf);
  set_options_from_list(option_reg_list, opt_reg);

  const Index n = yy.rows();
  const Index m = yy.cols();
  const Index K = opt_mf.k();

  auto mf_theta_u = make_dense_col_spike_slab<Scalar>(n, K, opt_mf);
  auto mf_theta_v = make_dense_col_spike_slab<Scalar>(m, K, opt_mf);
  auto mf_eta = make_factorization_eta(yy, mf_theta_u, mf_theta_v);

  if (opt_mf.mf_svd_init()) {
    mf_eta.init_by_svd(yy, opt_mf.jitter());
  } else {
    std::mt19937 rng(opt_mf.rseed());
    mf_eta.jitter(opt_mf.jitter(), rng);
  }

  auto mean_theta = make_dense_spike_slab<Scalar>(xx_mean.cols(), m, opt_reg);
  auto mean_eta = make_regression_eta(xx_mean, yy, mean_theta);

  auto var_theta = make_dense_col_slab<Scalar>(xx_var.cols(), m, opt_reg);
  auto var_eta = make_regression_eta(xx_var, yy, var_theta);

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();

  // Pre-train : Y ~ X model.  This will help convergence of
  // high-dimensional X.

  dummy_eta_t dummy;

  if (opt_mf.mf_pretrain() || n < xx_mean.cols()) {
    auto llik_pretrain = impl_fit_eta(model, opt_reg, std::make_tuple(mean_eta),
                                      std::make_tuple(var_eta));
    TLOG("Finished pre-training of regression");

    auto llik_clamped = impl_fit_eta(
        model, opt_mf, std::make_tuple(mf_eta), std::make_tuple(var_eta),
        std::make_tuple(mean_eta), std::make_tuple(dummy));
    TLOG("Finished pre-training of mf");
  }

  auto llik_trace =
      impl_fit_eta(model, opt_mf, std::make_tuple(mf_eta, mean_eta),
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
template <typename ModelTag>
Rcpp::List rcpp_train_mf_cis(
    const Mat &yy,          // n x m
    const Mat &xx_mean,     // n x p -> regression -> [n x p] [p x m]
    const SpMat &adj_mean,  // p x m adjacency
    const Mat &xx_var,      // n x q -> regression -> [n x q] [q x m]
    const Rcpp::List &option_mf_list, const Rcpp::List &option_reg_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");
  ASSERT_LIST_RET(xx_mean.cols() == adj_mean.rows(),
                  "xx_mean and adj_mean with different number");
  ASSERT_LIST_RET(yy.cols() == adj_mean.cols(),
                  "yy and adj_mean with different number of outputs");

  options_t opt_mf;
  options_t opt_reg;
  set_options_from_list(option_mf_list, opt_mf);
  set_options_from_list(option_reg_list, opt_reg);

  const Index n = yy.rows();
  const Index m = yy.cols();
  const Index K = opt_mf.k();

  auto mf_theta_u = make_dense_col_spike_slab<Scalar>(n, K, opt_mf);
  auto mf_theta_v = make_dense_col_spike_slab<Scalar>(m, K, opt_mf);
  auto mf_eta = make_factorization_eta(yy, mf_theta_u, mf_theta_v);
  if (opt_mf.mf_svd_init()) {
    mf_eta.init_by_svd(yy, opt_mf.jitter());
  } else {
    std::mt19937 rng(opt_mf.rseed());
    mf_eta.jitter(opt_mf.jitter(), rng);
  }

  auto mean_theta = make_sparse_spike_slab<Scalar>(adj_mean, opt_reg);
  auto mean_eta = make_regression_eta(xx_mean, yy, mean_theta);

  auto var_theta = make_dense_slab<Scalar>(xx_var.cols(), m, opt_reg);
  auto var_eta = make_regression_eta(xx_var, yy, var_theta);

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();

  // Pre-train : Y ~ X model.  This will help convergence of
  // high-dimensional X.

  dummy_eta_t dummy;

  if (opt_mf.mf_pretrain() || n < xx_mean.cols()) {
    auto llik_pretrain = impl_fit_eta(model, opt_reg, std::make_tuple(mean_eta),
                                      std::make_tuple(var_eta));
    TLOG("Finished pre-training");

    auto llik_clamped = impl_fit_eta(
        model, opt_mf, std::make_tuple(mf_eta), std::make_tuple(var_eta),
        std::make_tuple(mean_eta), std::make_tuple(dummy));
    TLOG("Finished pre-training of mf");
  }

  auto llik_trace =
      impl_fit_eta(model, opt_mf, std::make_tuple(mf_eta, mean_eta),
                   std::make_tuple(var_eta));

  TLOG("Finished MF");

  return Rcpp::List::create(Rcpp::_["U"] = param_rcpp_list(mf_theta_u),
                            Rcpp::_["V"] = param_rcpp_list(mf_theta_v),
                            Rcpp::_["mean"] = param_rcpp_list(mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(var_theta),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ U * V' + xx_sparse_mean * theta + xx_dense_mean *
// theta
// var ~ xx_var * theta
template <typename ModelTag>
Rcpp::List rcpp_train_mf_cis_aux(
    const Mat &yy,              // n x m
    const Mat &xx_sparse_mean,  // n x p
    const SpMat &adj_mean,      // p x m adjacency
    const Mat &xx_dense_mean,   // additional covariates
    const Mat &xx_var,          // n x q -> regression -> [n x q] [q x m]
    const Rcpp::List &option_mf_list, const Rcpp::List &option_reg_list) {
  //////////////////////
  // check dimensions //
  //////////////////////

  ASSERT_LIST_RET(yy.rows() == xx_sparse_mean.rows(),
                  "yy and xx_sparse_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");
  ASSERT_LIST_RET(xx_sparse_mean.cols() == adj_mean.rows(),
                  "xx_sparse_mean and adj_mean with different");
  ASSERT_LIST_RET(yy.cols() == adj_mean.cols(),
                  "yy and adj_mean with different number of outputs");

  options_t opt_mf;
  options_t opt_reg;
  set_options_from_list(option_mf_list, opt_mf);
  set_options_from_list(option_reg_list, opt_reg);

  const Index n = yy.rows();
  const Index m = yy.cols();
  const Index K = opt_mf.k();

  auto mf_theta_u = make_dense_col_spike_slab<Scalar>(n, K, opt_mf);
  auto mf_theta_v = make_dense_col_spike_slab<Scalar>(m, K, opt_mf);
  auto mf_eta = make_factorization_eta(yy, mf_theta_u, mf_theta_v);
  if (opt_mf.mf_svd_init()) {
    mf_eta.init_by_svd(yy, opt_mf.jitter());
  } else {
    std::mt19937 rng(opt_mf.rseed());
    mf_eta.jitter(opt_mf.jitter(), rng);
  }

  auto mean_sparse_theta = make_sparse_spike_slab<Scalar>(adj_mean, opt_reg);
  auto mean_sparse_eta =
      make_regression_eta(xx_sparse_mean, yy, mean_sparse_theta);

  auto mean_dense_theta =
      make_dense_spike_slab<Scalar>(xx_dense_mean.cols(), m, opt_reg);
  auto mean_dense_eta =
      make_regression_eta(xx_dense_mean, yy, mean_dense_theta);

  auto var_theta = make_dense_slab<Scalar>(xx_var.cols(), m, opt_reg);
  auto var_eta = make_regression_eta(xx_var, yy, var_theta);

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();

  // Pre-train : Y ~ X model.  This will help convergence of
  // high-dimensional X.

  dummy_eta_t dummy;

  if (opt_mf.mf_pretrain() || n < xx_sparse_mean.cols() ||
      n < xx_dense_mean.cols()) {
    auto llik_pretrain = impl_fit_eta(
        model, opt_reg, std::make_tuple(mean_sparse_eta, mean_dense_eta),
        std::make_tuple(var_eta));
    TLOG("Finished pre-training");

    auto llik_clamped = impl_fit_eta(
        model, opt_mf, std::make_tuple(mf_eta), std::make_tuple(var_eta),
        std::make_tuple(mean_sparse_eta, mean_dense_eta),
        std::make_tuple(dummy));
    TLOG("Finished pre-training of mf");
  }

  auto llik_trace = impl_fit_eta(
      model, opt_mf, std::make_tuple(mf_eta, mean_sparse_eta, mean_dense_eta),
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
template <typename ModelTag>
Rcpp::List rcpp_train_factored_regression(const Mat &yy,       // n x m
                                          const Mat &xx_mean,  // n x p -> p x m
                                          const Mat &cc_mean,  // n x p -> p x m
                                          const Mat &xx_var,   // n x p
                                          const Rcpp::List &option_list) {
  ///////////////////
  // check options //
  ///////////////////

  if (yy.cols() < 2) WLOG("Factored regression with 1 output");

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
  const Index K = std::min(static_cast<Index>(opt.k()), yy.cols());

  auto c_mean_theta =
      make_dense_spike_slab<Scalar>(cc_mean.cols(), yy.cols(), opt);
  auto x_var_theta = make_dense_slab<Scalar>(xx_var.cols(), yy.cols(), opt);

  auto c_mean_eta = make_regression_eta(cc_mean, yy, c_mean_theta);
  auto x_var_eta = make_regression_eta(xx_var, yy, x_var_theta);

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();
  dummy_eta_t dummy;
  auto theta_resid = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);

  Rcpp::List mf_left;
  Rcpp::List mf_right;
  Mat llik_trace;

  if (opt.mf_right_nn()) {
    /////////////////////////////////
    // non-negativity on the right //
    /////////////////////////////////

    auto mf_theta_u = make_dense_spike_slab<Scalar>(p, K, opt);
    auto mf_theta_v = make_dense_spike_gamma<Scalar>(m, K, opt);
    auto mean_eta =
        make_factored_regression_eta(xx_mean, yy, mf_theta_u, mf_theta_v);

    if (opt.mf_svd_init()) {
      mean_eta.init_by_svd(yy, opt.jitter());
    } else {
      std::mt19937 rng(opt.rseed());
      mean_eta.jitter(opt.jitter(), rng);
    }
    llik_trace = impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                              std::make_tuple(x_var_eta));

    // residual calculation
    if (opt.out_resid()) {
      auto resid_eta = make_residual_eta(yy, theta_resid);
      impl_fit_eta(
          model, opt, std::make_tuple(resid_eta), std::make_tuple(x_var_eta),
          std::make_tuple(mean_eta, c_mean_eta), std::make_tuple(dummy));
    }

    mf_left = param_rcpp_list(mf_theta_u);
    mf_right = param_rcpp_list(mf_theta_v);

  } else {
    //////////////////
    // regular FQTL //
    //////////////////

    auto mf_theta_u = make_dense_spike_slab<Scalar>(p, K, opt);
    auto mf_theta_v = make_dense_spike_slab<Scalar>(m, K, opt);
    auto mean_eta =
        make_factored_regression_eta(xx_mean, yy, mf_theta_u, mf_theta_v);

    if (opt.mf_svd_init()) {
      mean_eta.init_by_svd(yy, opt.jitter());
    } else {
      std::mt19937 rng(opt.rseed());
      mean_eta.jitter(opt.jitter(), rng);
    }
    llik_trace = impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                              std::make_tuple(x_var_eta));

    // residual calculation
    if (opt.out_resid()) {
      auto resid_eta = make_residual_eta(yy, theta_resid);
      impl_fit_eta(
          model, opt, std::make_tuple(resid_eta), std::make_tuple(x_var_eta),
          std::make_tuple(mean_eta, c_mean_eta), std::make_tuple(dummy));
    }

    mf_left = param_rcpp_list(mf_theta_u);
    mf_right = param_rcpp_list(mf_theta_v);
  }

  return Rcpp::List::create(Rcpp::_["mean.left"] = mf_left,
                            Rcpp::_["mean.right"] = mf_right,
                            Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(x_var_theta),
                            Rcpp::_["resid"] = param_rcpp_list(theta_resid),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ ((X * U) .* W) * V' + C * theta + theta * Ct
// var  ~ Xv * theta
template <typename ModelTag>
Rcpp::List rcpp_train_factored_regression(const Mat &yy,       // n x m
                                          const Mat &xx_mean,  // n x p -> p x m
                                          const Mat &cc_mean,  // n x p -> p x m
                                          const Mat &xx_var,   // n x p
                                          const Mat &weight,   // n x k
                                          const Rcpp::List &option_list) {
  ///////////////////
  // check options //
  ///////////////////

  if (yy.cols() < 2) WLOG("Factored regression with 1 output");

  ASSERT_LIST_RET(yy.rows() == xx_mean.rows(),
                  "yy and xx_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == cc_mean.rows(),
                  "yy and cc_mean with different number of rows");
  ASSERT_LIST_RET(yy.rows() == xx_var.rows(),
                  "yy and xx_var with different number of rows");
  ASSERT_LIST_RET(yy.rows() == weight.rows(),
                  "yy and weight with different number of rows");

  options_t opt;
  set_options_from_list(option_list, opt);

  //////////////////////////
  // construct parameters //
  //////////////////////////

  const Index p = xx_mean.cols();
  const Index m = yy.cols();
  const Index K = weight.cols();
  TLOG("K = " << K << " columns in the weight matrix");

  auto c_mean_theta =
      make_dense_spike_slab<Scalar>(cc_mean.cols(), yy.cols(), opt);
  auto x_var_theta = make_dense_slab<Scalar>(xx_var.cols(), yy.cols(), opt);

  auto c_mean_eta = make_regression_eta(cc_mean, yy, c_mean_theta);
  auto x_var_eta = make_regression_eta(xx_var, yy, x_var_theta);

  auto mf_theta_u = make_dense_spike_slab<Scalar>(p, K, opt);
  auto mf_theta_v = make_dense_beta<Scalar>(m, K, opt);
  auto mean_eta = make_factored_weighted_regression_eta(xx_mean, yy, mf_theta_u,
                                                        mf_theta_v);
  mean_eta.set_weight_nk(weight);

  if (opt.mf_svd_init()) {
    mean_eta.init_by_svd(yy, opt.jitter());
  } else {
    std::mt19937 rng(opt.rseed());
    mean_eta.jitter(opt.jitter(), rng);
  }

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  // residual calculation
  dummy_eta_t dummy;
  auto theta_resid = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  if (opt.out_resid()) {
    auto resid_eta = make_residual_eta(yy, theta_resid);
    impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                 std::make_tuple(x_var_eta),
                 std::make_tuple(mean_eta, c_mean_eta), std::make_tuple(dummy));
  }

  return Rcpp::List::create(Rcpp::_["mean.left"] = param_rcpp_list(mf_theta_u),
                            Rcpp::_["mean.right"] = param_rcpp_list(mf_theta_v),
                            Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(x_var_theta),
                            Rcpp::_["resid"] = param_rcpp_list(theta_resid),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * U * V' + C * theta
// var  ~ Xv * theta
template <typename ModelTag>
Rcpp::List rcpp_train_factored_regression_cis(
    const Mat &yy,             // n x m
    const Mat &xx_mean,        // n x p -> p x m
    const Mat &cc_mean,        // n x p -> p x m
    const SpMat &adj_cc_mean,  // p x m
    const Mat &xx_var,         // n x p
    const Rcpp::List &option_list) {
  ///////////////////
  // check options //
  ///////////////////

  if (yy.cols() < 2) WLOG("Factored regression with 1 output");

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
  const Index K = std::min(static_cast<Index>(opt.k()), yy.cols());

  auto c_mean_theta = make_sparse_spike_slab<Scalar>(adj_cc_mean, opt);
  auto x_var_theta = make_dense_slab<Scalar>(xx_var.cols(), yy.cols(), opt);

  auto c_mean_eta = make_regression_eta(cc_mean, yy, c_mean_theta);
  auto x_var_eta = make_regression_eta(xx_var, yy, x_var_theta);

  auto mf_theta_u = make_dense_spike_slab<Scalar>(p, K, opt);
  auto mf_theta_v = make_dense_spike_slab<Scalar>(m, K, opt);
  auto mean_eta =
      make_factored_regression_eta(xx_mean, yy, mf_theta_u, mf_theta_v);

  if (opt.mf_svd_init()) {
    mean_eta.init_by_svd(yy, opt.jitter());
  } else {
    std::mt19937 rng(opt.rseed());
    mean_eta.jitter(opt.jitter(), rng);
  }

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  // residual calculation
  dummy_eta_t dummy;
  auto theta_resid = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  if (opt.out_resid()) {
    auto resid_eta = make_residual_eta(yy, theta_resid);
    impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                 std::make_tuple(x_var_eta),
                 std::make_tuple(mean_eta, c_mean_eta), std::make_tuple(dummy));
  }

  return Rcpp::List::create(Rcpp::_["mean.left"] = param_rcpp_list(mf_theta_u),
                            Rcpp::_["mean.right"] = param_rcpp_list(mf_theta_v),
                            Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(x_var_theta),
                            Rcpp::_["resid"] = param_rcpp_list(theta_resid),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * theta + C * theta
// var  ~ Xv * theta
template <typename ModelTag>
Rcpp::List rcpp_train_regression(const Mat &yy,       // n x m
                                 const Mat &xx_mean,  // n x p
                                 const Mat &cc_mean,  // n x p
                                 const Mat &xx_var,   // n x p
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
  mean_eta.init_by_dot(yy, opt.jitter());

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  // residual calculation
  dummy_eta_t dummy;
  auto theta_resid = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  if (opt.out_resid()) {
    auto resid_eta = make_residual_eta(yy, theta_resid);
    impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                 std::make_tuple(x_var_eta),
                 std::make_tuple(mean_eta, c_mean_eta), std::make_tuple(dummy));
  }

  return Rcpp::List::create(Rcpp::_["mean"] = param_rcpp_list(mean_theta),
                            Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(x_var_theta),
                            Rcpp::_["resid"] = param_rcpp_list(theta_resid),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * theta + C * theta
// var  ~ Xv * theta
template <typename ModelTag>
Rcpp::List rcpp_train_regression_cis(const Mat &yy,          // n x m
                                     const Mat &xx_mean,     // n x p
                                     const SpMat &adj_mean,  // p x m
                                     const Mat &cc_mean,     // n x p
                                     const Mat &xx_var,      // n x p
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
                  "xx_mean and adj_mean with different number");
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

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  // residual calculation
  dummy_eta_t dummy;
  auto theta_resid = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  if (opt.out_resid()) {
    auto resid_eta = make_residual_eta(yy, theta_resid);
    impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                 std::make_tuple(x_var_eta),
                 std::make_tuple(mean_eta, c_mean_eta), std::make_tuple(dummy));
  }

  return Rcpp::List::create(Rcpp::_["mean"] = param_rcpp_list(mean_theta),
                            Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(x_var_theta),
                            Rcpp::_["resid"] = param_rcpp_list(theta_resid),
                            Rcpp::_["llik"] = llik_trace);
}

////////////////////////////////////////////////////////////////
// mean ~ X * theta + C * theta
// var  ~ Xv * theta
template <typename ModelTag>
Rcpp::List rcpp_train_regression_cis_cis(const Mat &yy,             // n x m
                                         const Mat &xx_mean,        // n x p
                                         const SpMat &adj_xx_mean,  // p x m
                                         const Mat &cc_mean,        // n x p
                                         const SpMat &adj_cc_mean,  // p x m
                                         const Mat &xx_var,         // n x p
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
                  "xx_mean and adj_xx_mean with different");
  ASSERT_LIST_RET(yy.cols() == adj_xx_mean.cols(),
                  "yy and adj_xx_mean with different number of outputs");
  ASSERT_LIST_RET(cc_mean.cols() == adj_cc_mean.rows(),
                  "cc_mean and adj_cc_mean with different");
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

  auto model_ptr = make_model<ModelTag>(yy);
  auto &model = *model_ptr.get();

  auto llik_trace =
      impl_fit_eta(model, opt, std::make_tuple(mean_eta, c_mean_eta),
                   std::make_tuple(x_var_eta));

  // residual calculation
  dummy_eta_t dummy;
  auto theta_resid = make_dense_col_slab<Scalar>(yy.rows(), yy.cols(), opt);
  if (opt.out_resid()) {
    auto resid_eta = make_residual_eta(yy, theta_resid);
    impl_fit_eta(model, opt, std::make_tuple(resid_eta),
                 std::make_tuple(x_var_eta),
                 std::make_tuple(mean_eta, c_mean_eta), std::make_tuple(dummy));
  }

  return Rcpp::List::create(Rcpp::_["mean"] = param_rcpp_list(mean_theta),
                            Rcpp::_["mean.cov"] = param_rcpp_list(c_mean_theta),
                            Rcpp::_["var"] = param_rcpp_list(x_var_theta),
                            Rcpp::_["resid"] = param_rcpp_list(theta_resid),
                            Rcpp::_["llik"] = llik_trace);
}

template <typename T>
Rcpp::List param_rcpp_list(const T &param) {
  return impl_param_rcpp_list(param, sgd_tag<T>());
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_spike_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_spike_gamma) {
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

template <typename T>
Rcpp::List impl_param_rcpp_list(const T &param, const tag_param_beta) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param),
                            Rcpp::_["theta.var"] = var_param(param));
}

///////////////
// Utilities //
///////////////

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
  if (_list.containsElementNamed("vb.tol"))
    opt.VBTOL = Rcpp::as<Scalar>(_list["vb.tol"]);
  if (_list.containsElementNamed("k")) opt.K = Rcpp::as<Index>(_list["k"]);
  if (_list.containsElementNamed("K")) opt.K = Rcpp::as<Index>(_list["K"]);
  if (_list.containsElementNamed("gammax"))
    opt.GAMMAX = Rcpp::as<Scalar>(_list["gammax"]);
  if (_list.containsElementNamed("decay"))
    opt.DECAY = Rcpp::as<Scalar>(_list["decay"]);
  if (_list.containsElementNamed("rate"))
    opt.RATE0 = Rcpp::as<Scalar>(_list["rate"]);
  if (_list.containsElementNamed("nsample")) {
    opt.NSAMPLE = Rcpp::as<Index>(_list["nsample"]);
    if (opt.nsample() < 3)
      WLOG("Too small random samples in SGD : " << opt.nsample());
  }
  if (_list.containsElementNamed("adam.rate.m"))
    opt.RATE_M = Rcpp::as<Scalar>(_list["adam.rate.m"]);
  if (_list.containsElementNamed("adam.rate.v"))
    opt.RATE_V = Rcpp::as<Scalar>(_list["adam.rate.v"]);
  if (_list.containsElementNamed("rseed"))
    opt.RSEED = Rcpp::as<Index>(_list["rseed"]);
  if (_list.containsElementNamed("jitter"))
    opt.JITTER = Rcpp::as<Scalar>(_list["jitter"]);
  if (_list.containsElementNamed("svd.init"))
    opt.MF_SVD_INIT = Rcpp::as<bool>(_list["svd.init"]);

  if (_list.containsElementNamed("mf.pretrain"))
    opt.MF_PRETRAIN = Rcpp::as<bool>(_list["mf.pretrain"]);

  if (_list.containsElementNamed("mf.right.nn"))
    opt.MF_RIGHT_NN = Rcpp::as<bool>(_list["mf.right.nn"]);

  if (_list.containsElementNamed("vbiter"))
    opt.VBITER = Rcpp::as<Index>(_list["vbiter"]);

  if (_list.containsElementNamed("print.interv"))
    opt.INTERV = Rcpp::as<Index>(_list["print.interv"]);

  if (_list.containsElementNamed("num.threads"))
    opt.NTHREAD = Rcpp::as<Index>(_list["num.threads"]);

  if (_list.containsElementNamed("nthreads"))
    opt.NTHREAD = Rcpp::as<Index>(_list["nthreads"]);

  if (_list.containsElementNamed("verbose"))
    opt.VERBOSE = Rcpp::as<bool>(_list["verbose"]);
  if (_list.containsElementNamed("out.residual"))
    opt.OUT_RESID = Rcpp::as<bool>(_list["out.residual"]);
  if (_list.containsElementNamed("model"))
    opt.MODEL_NAME = Rcpp::as<std::string>(_list["model"]);

  if (_list.containsElementNamed("do.hyper"))
    opt.DO_HYPER = Rcpp::as<bool>(_list["do.hyper"]);
  if (_list.containsElementNamed("tau")) {
    opt.TAU_LODDS_LB = Rcpp::as<Scalar>(_list["tau"]);
    opt.TAU_LODDS_UB = Rcpp::as<Scalar>(_list["tau"]);
  }
  if (_list.containsElementNamed("pi")) {
    opt.PI_LODDS_LB = Rcpp::as<Scalar>(_list["pi"]);
    opt.PI_LODDS_UB = Rcpp::as<Scalar>(_list["pi"]);
  }
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
      if (d2_start > d2_end) continue;
      if (d1 >= (d2_start - cis_window) && d1 <= (d2_end + cis_window)) {
        left.push_back(i + 1);
        right.push_back(j + 1);
      }
    }
  }

  return Rcpp::List::create(Rcpp::_["d1"] = Rcpp::wrap(left),
                            Rcpp::_["d2"] = Rcpp::wrap(right));
}
