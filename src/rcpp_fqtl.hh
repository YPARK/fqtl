// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "convergence.hh"
#include "dummy.hh"
#include "factored_regression.hh"
#include "factorization.hh"
#include "gaussian.hh"
#include "gaussian_voom.hh"
#include "options.hh"
#include "parameters.hh"
#include "rcpp_util.hh"
#include "regression.hh"
#include "residual.hh"
#include "shared_effect.hh"

#ifndef RCPP_FQT_HH_
#define RCPP_FQT_HH_

using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor>;
using Scalar = Mat::Scalar;
using Index = Mat::Index;

////////////////////////////////////////////////////////////////
// Flags for hyperparameter tuning
struct do_hyper_tune {};
struct dont_hyper_tune {};
template <typename T>
struct ask_hyper_training {
  static constexpr bool value = false;
};
template <>
struct ask_hyper_training<do_hyper_tune> {
  static constexpr bool value = true;
};

struct do_stoch_tune {};
struct dont_stoch_tune {};
template <typename T>
struct ask_stoch_training {
  static constexpr bool value = false;
};
template <>
struct ask_stoch_training<do_stoch_tune> {
  static constexpr bool value = true;
};

////////////////////////////////////////////////////////////////
// fit model E[Y] ~ f(mean1 + mean2 + mean3 + mean4 + mean5)
//           V[Y] ~ g(var1 + var2 + var3)
template <typename Stoch, typename EtaMean1, typename EtaMean2, typename EtaMean3, typename EtaMean4, typename EtaMean5,
          typename EtaVar1, typename EtaVar2, typename EtaVar3>
auto fit_model(const std::string model_name, const Mat& Y, EtaMean1& mean_eta1, EtaMean2& mean_eta2,
               EtaMean3& mean_eta3, EtaMean4& mean_eta4, EtaMean5& mean_eta5, EtaVar1& var_eta1, EtaVar2& var_eta2,
               EtaVar3& var_eta3, const options_t& opt);

// fit residuals
template <typename Stoch, typename EtaResid, typename EtaMean1, typename EtaMean2, typename EtaMean3, typename EtaMean4,
          typename EtaMean5, typename EtaVar1, typename EtaVar2, typename EtaVar3>
auto fit_residual(const std::string model_name, const Mat& Y, EtaResid& resid_eta, EtaMean1& mean_eta1,
                  EtaMean2& mean_eta2, EtaMean3& mean_eta3, EtaMean4& mean_eta4, EtaMean5& mean_eta5, EtaVar1& var_eta1,
                  EtaVar2& var_eta2, EtaVar3& var_eta3, const options_t& opt);

/////////////////////////////
// Update eta of the model //
/////////////////////////////

template <typename Hyper, typename Stoch, typename UpdateEta, typename Model, typename MeanFixed1, typename MeanFixed2,
          typename MeanFixed3, typename MeanFixed4, typename VarFixed1, typename VarFixed2, typename VarFixed3,
          typename RnormOp>
inline void update_mean_eta(UpdateEta& eta_to_update, Model& model, MeanFixed1& fixed_eta1, MeanFixed2& fixed_eta2,
                            MeanFixed3& fixed_eta3, MeanFixed4& fixed_eta4, const Index S, const Scalar rate,
                            VarFixed1& var_fixed_eta1, VarFixed2& var_fixed_eta2, VarFixed3& var_fixed_eta3,
                            RnormOp& rnorm_op) {
  for (Index s = 0; s < S; ++s) {
    if (ask_stoch_training<Stoch>::value) {
      model.eval(eta_to_update.sample(rnorm_op) + fixed_eta1.sample(rnorm_op) + fixed_eta2.sample(rnorm_op) +
                     fixed_eta3.sample(rnorm_op) + fixed_eta4.sample(rnorm_op),
                 var_fixed_eta1.sample(rnorm_op) + var_fixed_eta2.sample(rnorm_op) + var_fixed_eta3.sample(rnorm_op));
    } else {
      model.eval(eta_to_update.sample(rnorm_op) + fixed_eta1.repr_mean() + fixed_eta2.repr_mean() +
                     fixed_eta3.repr_mean() + fixed_eta4.repr_mean(),
                 var_fixed_eta1.repr_mean() + var_fixed_eta2.repr_mean() + var_fixed_eta3.repr_mean());
    }

    eta_to_update.add_sgd(model.llik());
  }
  eta_to_update.eval_sgd();
  eta_to_update.update_sgd(rate);
  if (ask_hyper_training<Hyper>::value) {
    eta_to_update.eval_hyper_sgd();
    eta_to_update.update_hyper_sgd(rate);
  }
}

template <typename Hyper, typename Stoch, typename Model, typename MeanFixed1, typename MeanFixed2, typename MeanFixed3,
          typename MeanFixed4, typename VarFixed1, typename VarFixed2, typename VarFixed3, typename RnormOp>
inline void update_mean_eta(dummy_eta_t& eta_to_update, Model& model, MeanFixed1& fixed_eta1, MeanFixed2& fixed_eta2,
                            MeanFixed3& fixed_eta3, MeanFixed4& fixed_eta4, const Index S, const Scalar rate,
                            VarFixed1& var_fixed_eta1, VarFixed2& var_fixed_eta2, VarFixed3& var_fixed_eta3,
                            RnormOp& rnorm_op) {}

template <typename Hyper, typename Stoch, typename UpdateEta, typename Model, typename MeanFixed1, typename MeanFixed2,
          typename MeanFixed3, typename MeanFixed4, typename VarFixed1, typename VarFixed2, typename VarFixed3,
          typename RnormOp>
inline void update_resid_eta(UpdateEta& eta_to_update, Model& model, MeanFixed1& fixed_eta1, MeanFixed2& fixed_eta2,
                             MeanFixed3& fixed_eta3, MeanFixed4& fixed_eta4, MeanFixed4& fixed_eta5, const Index S,
                             const Scalar rate, VarFixed1& var_fixed_eta1, VarFixed2& var_fixed_eta2,
                             VarFixed3& var_fixed_eta3, RnormOp& rnorm_op) {
  for (Index s = 0; s < S; ++s) {
    if (ask_stoch_training<Stoch>::value) {
      model.eval(eta_to_update.sample(rnorm_op) + fixed_eta1.sample(rnorm_op) + fixed_eta2.sample(rnorm_op) +
                     fixed_eta3.sample(rnorm_op) + fixed_eta4.sample(rnorm_op) + fixed_eta5.sample(rnorm_op),
                 var_fixed_eta1.sample(rnorm_op) + var_fixed_eta2.sample(rnorm_op) + var_fixed_eta3.sample(rnorm_op));
    } else {
      model.eval(eta_to_update.sample(rnorm_op) + fixed_eta1.repr_mean() + fixed_eta2.repr_mean() +
                     fixed_eta3.repr_mean() + fixed_eta4.repr_mean() + fixed_eta5.repr_mean(),
                 var_fixed_eta1.repr_mean() + var_fixed_eta2.repr_mean() + var_fixed_eta3.repr_mean());
    }

    eta_to_update.add_sgd(model.llik());
  }
  eta_to_update.eval_sgd();
  eta_to_update.update_sgd(rate);
  if (ask_hyper_training<Hyper>::value) {
    eta_to_update.eval_hyper_sgd();
    eta_to_update.update_hyper_sgd(rate);
  }
}

///////////////////////////////////
// Update eta of the model's var //
///////////////////////////////////

template <typename Hyper, typename Stoch, typename Model, typename UpdateEta, typename MeanFixed1, typename MeanFixed2,
          typename MeanFixed3, typename MeanFixed4, typename MeanFixed5, typename VarFixed1, typename VarFixed2,
          typename RnormOp>
inline void update_var_eta(UpdateEta& var_eta_to_update, Model& model, MeanFixed1& fixed_eta1, MeanFixed2& fixed_eta2,
                           MeanFixed3& fixed_eta3, MeanFixed4& fixed_eta4, MeanFixed5& fixed_eta5, const Index S,
                           const Scalar rate, VarFixed1& var_fixed_eta1, VarFixed2& var_fixed_eta2, RnormOp& rnorm_op) {
  for (Index s = 0; s < S; ++s) {
    if (ask_stoch_training<Stoch>::value) {
      model.eval(
          fixed_eta1.sample(rnorm_op) + fixed_eta2.sample(rnorm_op) + fixed_eta3.sample(rnorm_op) +
              fixed_eta4.sample(rnorm_op) + fixed_eta5.sample(rnorm_op),
          var_eta_to_update.sample(rnorm_op) + var_fixed_eta1.sample(rnorm_op) + var_fixed_eta2.sample(rnorm_op));
    } else {
      model.eval(fixed_eta1.repr_mean() + fixed_eta2.repr_mean() + fixed_eta3.repr_mean() + fixed_eta4.repr_mean() +
                     fixed_eta5.repr_mean(),
                 var_eta_to_update.sample(rnorm_op) + var_fixed_eta1.repr_mean() + var_fixed_eta2.repr_mean());
    }

    var_eta_to_update.add_sgd(model.llik());
  }
  var_eta_to_update.eval_sgd();
  var_eta_to_update.update_sgd(rate);
  if (ask_hyper_training<Hyper>::value) {
    var_eta_to_update.eval_hyper_sgd();
    var_eta_to_update.update_hyper_sgd(rate);
  }
}

template <typename Hyper, typename Stoch, typename Model, typename MeanFixed1, typename MeanFixed2, typename MeanFixed3,
          typename MeanFixed4, typename MeanFixed5, typename VarFixed1, typename VarFixed2, typename RnormOp>
inline void update_var_eta(dummy_eta_t& var_eta_to_update, Model& model, MeanFixed1& fixed_eta1, MeanFixed2& fixed_eta2,
                           MeanFixed3& fixed_eta3, MeanFixed4& fixed_eta4, MeanFixed5& fixed_eta5, const Index S,
                           const Scalar rate, VarFixed1& var_fixed_eta1, VarFixed2& var_fixed_eta2, RnormOp& rnorm_op) {
}

///////////////////
// Iterative Alg //
///////////////////

template <typename Stoch, typename Model, typename EtaMean1, typename EtaMean2, typename EtaMean3, typename EtaMean4,
          typename EtaMean5, typename EtaVar1, typename EtaVar2, typename EtaVar3>
auto impl_fit_model(Model& model, EtaMean1& mean_eta1, EtaMean2& mean_eta2, EtaMean3& mean_eta3, EtaMean4& mean_eta4,
                    EtaMean5& mean_eta5, EtaVar1& var_eta1, EtaVar2& var_eta2, EtaVar3& var_eta3,
                    const options_t& opt) {
  // C++ random number generator and normal distribution were faster
  // than R's one
  std::mt19937 rng(std::time(0));
  std::normal_distribution<Scalar> Norm;
  auto rnorm_op = [&rng, &Norm] { return Norm(rng); };

  const Index S = opt.nsample();
  const Index n = model.n;
  const Index m = model.m;
  const Index ninterv = opt.ninterval();
  Mat onesN = Mat::Ones(n, 1) / static_cast<Scalar>(n);
  using conv_t = convergence_t<Scalar>;
  conv_t conv((conv_t::Nmodels(m)), (conv_t::Interv(ninterv)));
  Index t;

  // Must keep this progress obj; otherwise segfault will occur
  Progress prog(2 * opt.vbiter(), !opt.verbose());

  for (t = 0; t < opt.vbiter(); ++t) {
    const Scalar rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());

    if (Progress::check_abort()) {
      break;
    }
    prog.increment();

    update_mean_eta<dont_hyper_tune, Stoch>(mean_eta1, model, mean_eta2, mean_eta3, mean_eta4, mean_eta5, S, rate,
                                            var_eta1, var_eta2, var_eta3, rnorm_op);

    update_mean_eta<dont_hyper_tune, Stoch>(mean_eta2, model, mean_eta1, mean_eta3, mean_eta4, mean_eta5, S, rate,
                                            var_eta1, var_eta2, var_eta3, rnorm_op);

    update_mean_eta<dont_hyper_tune, Stoch>(mean_eta3, model, mean_eta1, mean_eta2, mean_eta4, mean_eta5, S, rate,
                                            var_eta1, var_eta2, var_eta3, rnorm_op);

    update_mean_eta<dont_hyper_tune, Stoch>(mean_eta4, model, mean_eta1, mean_eta2, mean_eta3, mean_eta5, S, rate,
                                            var_eta1, var_eta2, var_eta3, rnorm_op);

    update_mean_eta<dont_hyper_tune, Stoch>(mean_eta5, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4, S, rate,
                                            var_eta1, var_eta2, var_eta3, rnorm_op);

    update_var_eta<dont_hyper_tune, Stoch>(var_eta1, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4, mean_eta5, S,
                                           rate, var_eta2, var_eta3, rnorm_op);

    update_var_eta<dont_hyper_tune, Stoch>(var_eta2, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4, mean_eta5, S,
                                           rate, var_eta1, var_eta3, rnorm_op);

    update_var_eta<dont_hyper_tune, Stoch>(var_eta3, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4, mean_eta5, S,
                                           rate, var_eta1, var_eta2, rnorm_op);

    // check convergence
    model.eval(mean_eta1.repr_mean() + mean_eta2.repr_mean() + mean_eta3.repr_mean() + mean_eta4.repr_mean() +
                   mean_eta5.repr_mean(),
               var_eta1.repr_mean() + var_eta2.repr_mean() + var_eta3.repr_mean());
    conv.add(model.llik().transpose() * onesN);
    bool converged = conv.converged(opt.vbtol(), opt.miniter());
    if (opt.verbose()) conv.print(Rcpp::Rcerr);
    if (converged) {
      TLOG("converged log-likelihood");
      break;
    }
  }

  const Index t_stop = t;

  // Hyper-parameter tuning
  for (; t < 2 * opt.vbiter(); ++t) {
    const Scalar rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());

    if (Progress::check_abort()) {
      break;
    }
    prog.increment();

    update_mean_eta<do_hyper_tune, dont_stoch_tune>(mean_eta1, model, mean_eta2, mean_eta3, mean_eta4, mean_eta5, S,
                                                    rate, var_eta1, var_eta2, var_eta3, rnorm_op);

    update_mean_eta<do_hyper_tune, dont_stoch_tune>(mean_eta2, model, mean_eta1, mean_eta3, mean_eta4, mean_eta5, S,
                                                    rate, var_eta1, var_eta2, var_eta3, rnorm_op);

    update_mean_eta<do_hyper_tune, dont_stoch_tune>(mean_eta3, model, mean_eta1, mean_eta2, mean_eta4, mean_eta5, S,
                                                    rate, var_eta1, var_eta2, var_eta3, rnorm_op);

    update_mean_eta<do_hyper_tune, dont_stoch_tune>(mean_eta4, model, mean_eta1, mean_eta2, mean_eta3, mean_eta5, S,
                                                    rate, var_eta1, var_eta2, var_eta3, rnorm_op);

    update_mean_eta<do_hyper_tune, dont_stoch_tune>(mean_eta5, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4, S,
                                                    rate, var_eta1, var_eta2, var_eta3, rnorm_op);

    update_var_eta<do_hyper_tune, dont_stoch_tune>(var_eta1, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4,
                                                   mean_eta5, S, rate, var_eta2, var_eta3, rnorm_op);

    update_var_eta<do_hyper_tune, dont_stoch_tune>(var_eta2, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4,
                                                   mean_eta5, S, rate, var_eta1, var_eta3, rnorm_op);

    update_var_eta<do_hyper_tune, dont_stoch_tune>(var_eta3, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4,
                                                   mean_eta5, S, rate, var_eta1, var_eta2, rnorm_op);

    // check convergence
    model.eval(mean_eta1.repr_mean() + mean_eta2.repr_mean() + mean_eta3.repr_mean() + mean_eta4.repr_mean() +
                   mean_eta5.repr_mean(),
               var_eta1.repr_mean() + var_eta2.repr_mean() + var_eta3.repr_mean());
    conv.add(model.llik().transpose() * onesN);
    bool converged = conv.converged(opt.vbtol(), opt.miniter() + t_stop);
    if (opt.verbose()) conv.print(Rcpp::Rcerr);
    if (converged) {
      TLOG("converged log-likelihood");
      break;
    }
  }
  prog.increment();

  Mat ret = conv.summarize();
  return ret;
}

////////////////////////////////////////////////////////////////

template <typename Stoch, typename Model, typename EtaResid, typename EtaMean1, typename EtaMean2, typename EtaMean3,
          typename EtaMean4, typename EtaMean5, typename EtaVar1, typename EtaVar2, typename EtaVar3>
auto impl_fit_residual(Model& model, EtaResid& resid_eta, EtaMean1& mean_eta1, EtaMean2& mean_eta2, EtaMean3& mean_eta3,
                       EtaMean4& mean_eta4, EtaMean5& mean_eta5, EtaVar1& var_eta1, EtaVar2& var_eta2,
                       EtaVar3& var_eta3, const options_t& opt) {
  // C++ random number generator and normal distribution were faster
  // than R's one
  std::mt19937 rng(std::time(0));
  std::normal_distribution<Scalar> Norm;
  auto rnorm_op = [&rng, &Norm] { return Norm(rng); };

  const Index S = opt.nsample();
  const Index n = model.n;
  const Index m = model.m;
  const Index ninterv = opt.ninterval();
  Mat onesN = Mat::Ones(n, 1) / static_cast<Scalar>(n);
  using conv_t = convergence_t<Scalar>;
  conv_t conv((conv_t::Nmodels(m)), (conv_t::Interv(ninterv)));
  Index t;

  // Must keep this progress obj; otherwise segfault will occur
  Progress prog(2 * opt.vbiter(), !opt.verbose());

  for (t = 0; t < opt.vbiter(); ++t) {
    const Scalar rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());

    if (Progress::check_abort()) {
      break;
    }
    prog.increment();

    update_resid_eta<dont_hyper_tune, Stoch>(resid_eta, model, mean_eta1, mean_eta2, mean_eta3, mean_eta4, mean_eta5, S,
                                             rate, var_eta1, var_eta2, var_eta3, rnorm_op);

    // check convergence
    model.eval(resid_eta.repr_mean() + mean_eta1.repr_mean() + mean_eta2.repr_mean() + mean_eta3.repr_mean() +
                   mean_eta4.repr_mean() + mean_eta5.repr_mean(),
               var_eta1.repr_mean() + var_eta2.repr_mean() + var_eta3.repr_mean());
    conv.add(model.llik().transpose() * onesN);
    bool converged = conv.converged(opt.vbtol(), opt.miniter());
    if (opt.verbose()) conv.print(Rcpp::Rcerr);
    if (converged) {
      TLOG("converged log-likelihood");
      break;
    }
  }

  prog.increment();

  Mat ret = conv.summarize();
  return ret;
}

////////////////////////////////////////////////////////////////

template <typename Derived, typename OtherDerived>
void summarize_llik(const Eigen::MatrixBase<Derived>& llik, Eigen::MatrixBase<OtherDerived>& llik_trace) {
  llik_trace.resize(llik.rows(), 1);
  llik_trace.derived() = llik.rowwise().mean();
}

template <typename Stoch, typename EtaMean1, typename EtaMean2, typename EtaMean3, typename EtaMean4, typename EtaMean5,
          typename EtaVar1, typename EtaVar2, typename EtaVar3>
auto fit_model(const std::string model_name, const Mat& Y, EtaMean1& mean_eta1, EtaMean2& mean_eta2,
               EtaMean3& mean_eta3, EtaMean4& mean_eta4, EtaMean5& mean_eta5, EtaVar1& var_eta1, EtaVar2& var_eta2,
               EtaVar3& var_eta3, const options_t& opt) {
  Mat llik_trace;

  calc_stat_t<Scalar> stat_op;
  visit(Y, stat_op);

  if (model_name == "gaussian") {
    using model_type = gaussian_model_t<Mat>;
    const Scalar vmax = stat_op.var();
    const Scalar vmin = vmax * 1e-4;
    TLOG("Gaussian model : Vmax " << vmax << ", Vmin " << vmin);
    model_type model(Y, model_type::Vmin_t(vmin), model_type::Vmax_t(vmax));
    auto llik = impl_fit_model<Stoch>(model, mean_eta1, mean_eta2, mean_eta3, mean_eta4, mean_eta5, var_eta1, var_eta2,
                                      var_eta3, opt);
    summarize_llik(llik, llik_trace);

  } else if (model_name == "gaussian.voom") {
    using model_type = gaussian_voom_model_t<Mat>;
    const Scalar vmax = stat_op.var();
    const Scalar vmin = vmax * 1e-4;
    TLOG("Gaussian model : Vmax " << vmax << ", Vmin " << vmin);
    model_type model(Y, model_type::Vmin_t(vmin), model_type::Vmax_t(vmax));
    auto llik = impl_fit_model<Stoch>(model, mean_eta1, mean_eta2, mean_eta3, mean_eta4, mean_eta5, var_eta1, var_eta2,
                                      var_eta3, opt);
    summarize_llik(llik, llik_trace);

  } else {
    ELOG("Not implemented : " << model_name);
  }

  return llik_trace;
}

template <typename Stoch, typename EtaResid, typename EtaMean1, typename EtaMean2, typename EtaMean3, typename EtaMean4,
          typename EtaMean5, typename EtaVar1, typename EtaVar2, typename EtaVar3>
auto fit_residual(const std::string model_name, const Mat& Y, EtaResid& resid_eta, EtaMean1& mean_eta1,
                  EtaMean2& mean_eta2, EtaMean3& mean_eta3, EtaMean4& mean_eta4, EtaMean5& mean_eta5, EtaVar1& var_eta1,
                  EtaVar2& var_eta2, EtaVar3& var_eta3, const options_t& opt) {
  Mat llik_trace;

  calc_stat_t<Scalar> stat_op;
  visit(Y, stat_op);

  if (model_name == "gaussian") {
    using model_type = gaussian_model_t<Mat>;
    const Scalar vmax = stat_op.var();
    const Scalar vmin = vmax * 1e-4;
    TLOG("Gaussian model : Vmax " << vmax << ", Vmin " << vmin);
    model_type model(Y, model_type::Vmin_t(vmin), model_type::Vmax_t(vmax));
    auto llik = impl_fit_residual<Stoch>(model, resid_eta, mean_eta1, mean_eta2, mean_eta3, mean_eta4, mean_eta5,
                                         var_eta1, var_eta2, var_eta3, opt);
    summarize_llik(llik, llik_trace);

  } else if (model_name == "gaussian.voom") {
    using model_type = gaussian_voom_model_t<Mat>;
    const Scalar vmax = stat_op.var();
    const Scalar vmin = vmax * 1e-4;
    TLOG("Gaussian model : Vmax " << vmax << ", Vmin " << vmin);
    model_type model(Y, model_type::Vmin_t(vmin), model_type::Vmax_t(vmax));
    auto llik = impl_fit_residual<Stoch>(model, resid_eta, mean_eta1, mean_eta2, mean_eta3, mean_eta4, mean_eta5,
                                         var_eta1, var_eta2, var_eta3, opt);
    summarize_llik(llik, llik_trace);

  } else {
    ELOG("Not implemented : " << model_name);
  }

  return llik_trace;
}

////////////////////////////////////////////////////////////////
void set_options_from_list(const Rcpp::List& _list, options_t& opt) {
  if (_list.containsElementNamed("tau.lb")) opt.TAU_LODDS_LB = Rcpp::as<Scalar>(_list["tau.lb"]);
  if (_list.containsElementNamed("tau.ub")) opt.TAU_LODDS_UB = Rcpp::as<Scalar>(_list["tau.ub"]);
  if (_list.containsElementNamed("pi.lb")) opt.PI_LODDS_LB = Rcpp::as<Scalar>(_list["pi.lb"]);
  if (_list.containsElementNamed("pi.ub")) opt.PI_LODDS_UB = Rcpp::as<Scalar>(_list["pi.ub"]);
  if (_list.containsElementNamed("tol")) opt.VBTOL = Rcpp::as<Scalar>(_list["tol"]);
  if (_list.containsElementNamed("k")) opt.K = Rcpp::as<Scalar>(_list["k"]);
  if (_list.containsElementNamed("K")) opt.K = Rcpp::as<Scalar>(_list["K"]);
  if (_list.containsElementNamed("gammax")) opt.GAMMAX = Rcpp::as<Scalar>(_list["gammax"]);
  if (_list.containsElementNamed("decay")) opt.DECAY = Rcpp::as<Scalar>(_list["decay"]);
  if (_list.containsElementNamed("rate")) opt.RATE0 = Rcpp::as<Scalar>(_list["rate"]);
  if (_list.containsElementNamed("vbiter")) opt.VBITER = Rcpp::as<Index>(_list["vbiter"]);
  if (_list.containsElementNamed("verbose")) opt.VERBOSE = Rcpp::as<bool>(_list["verbose"]);
  if (_list.containsElementNamed("out.residual")) opt.OUT_RESID = Rcpp::as<bool>(_list["out.residual"]);
  if (_list.containsElementNamed("model")) opt.MODEL_NAME = Rcpp::as<std::string>(_list["model"]);
}

////////////////////////////////////////////////////////////////
Rcpp::List rcpp_fqtl_adj_list(const Rcpp::NumericVector& d1_loc, const Rcpp::NumericVector& d2_start_loc,
                              const Rcpp::NumericVector& d2_end_loc, const double cis_window) {
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

  return Rcpp::List::create(Rcpp::_["d1"] = Rcpp::wrap(left), Rcpp::_["d2"] = Rcpp::wrap(right));
}

////////////////////////////////////////////////////////////////
template <typename T>
Rcpp::List param_rcpp_list(const T& param) {
  return impl_param_rcpp_list(param, sgd_tag<T>());
}


template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_spike_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param), Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_col_spike_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param), Rcpp::_["theta.var"] = var_param(param),
                            Rcpp::_["lodds"] = log_odds_param(param));
}

template <typename T>
Rcpp::List impl_param_rcpp_list(const T& param, const tag_param_slab) {
  return Rcpp::List::create(Rcpp::_["theta"] = mean_param(param), Rcpp::_["theta.var"] = var_param(param));
}

#endif
