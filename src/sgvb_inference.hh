// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include "convergence.hh"
#include "tuple_util.hh"

#ifndef SGVB_INFERENCE_HH_
#define SGVB_INFERENCE_HH_

// Fit multiple eta's
template <typename Model, typename Opt, typename... MeanEtas,
          typename... VarEtas, typename... ClampedMeanEtas>
auto impl_fit_eta(Model &model, const Opt &opt,
                  std::tuple<MeanEtas...> &&mean_eta_tup,
                  std::tuple<VarEtas...> &&var_eta_tup,
                  std::tuple<ClampedMeanEtas...> &&clamped_mean_eta_tup) {

  using Scalar = typename Model::Scalar;
  using Index = typename Model::Index;
  using Mat = typename Model::Data;

  // random seed initialization
  std::mt19937 rng(opt.rseed());
  std::normal_distribution<Scalar> Norm;
  auto rnorm_op = [&rng, &Norm] { return Norm(rng); };

  using conv_t = convergence_t<Scalar>;
  Mat onesN = Mat::Ones(model.n, 1) / static_cast<Scalar>(model.n);
  conv_t conv(typename conv_t::Nmodels(model.m),
              typename conv_t::Interv(opt.ninterval()));

  const Index nstoch = opt.nsample();
  const Index niter = opt.vbiter();
  Index t;

  Mat mean_sampled(model.n, model.m);
  Mat var_sampled(model.n, model.m);

  // Must keep this progress obj; otherwise segfault will occur
  Progress prog(2 * opt.vbiter(), !opt.verbose());

  // model fitting
  Scalar rate = opt.rate0();

  auto sample_mean_eta = [&rnorm_op, &mean_sampled](auto &&eta) {
    mean_sampled += eta.sample(rnorm_op);
  };

  auto sample_var_eta = [&rnorm_op, &var_sampled](auto &&eta) {
    var_sampled += eta.sample(rnorm_op);
  };

  auto update_sgd_eta = [&nstoch, &mean_eta_tup, &var_eta_tup,
                         &clamped_mean_eta_tup, &sample_mean_eta,
                         &sample_var_eta, &mean_sampled, &var_sampled, &model,
                         &rate](auto &&eta) {

    for (Index s = 0; s < nstoch; ++s) {
      mean_sampled.setZero();
      var_sampled.setZero();
      func_apply(sample_mean_eta, std::move(mean_eta_tup));
      func_apply(sample_var_eta, std::move(var_eta_tup));
      func_apply(sample_mean_eta, std::move(clamped_mean_eta_tup));
      model.eval(mean_sampled, var_sampled);
      eta.add_sgd(model.llik());
    }
    eta.eval_sgd();
    eta.update_sgd(rate);
  };

  auto update_hyper_sgd_eta = [&nstoch, &mean_eta_tup, &var_eta_tup,
                               &clamped_mean_eta_tup, &sample_mean_eta,
                               &sample_var_eta, &mean_sampled, &var_sampled,
                               &model, &rate](auto &&eta) {

    for (Index s = 0; s < nstoch; ++s) {
      mean_sampled.setZero();
      var_sampled.setZero();
      func_apply(sample_mean_eta, std::move(mean_eta_tup));
      func_apply(sample_var_eta, std::move(var_eta_tup));
      func_apply(sample_mean_eta, std::move(clamped_mean_eta_tup));
      model.eval(mean_sampled, var_sampled);
      eta.add_sgd(model.llik());
    }
    eta.eval_hyper_sgd();
    eta.update_hyper_sgd(rate);
    eta.eval_sgd();
    eta.update_sgd(rate);
  };

  // initial tuning without hyperparameter optimization
  for (t = 0; t < niter; ++t) {
    if (Progress::check_abort()) {
      break;
    }
    prog.increment();
    rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());
    func_apply(update_sgd_eta, std::move(mean_eta_tup));
    func_apply(update_sgd_eta, std::move(var_eta_tup));

    conv.add(model.llik().transpose() * onesN);
    bool converged = conv.converged(opt.vbtol(), opt.miniter());
    if (opt.verbose())
      conv.print(Rcpp::Rcerr);
    if (converged) {
      TLOG("Converged initial log-likelihood");
      break;
    }
  }

  // hyperparameter tuning
  for (; t < 2 * niter; ++t) {
    if (Progress::check_abort()) {
      break;
    }
    prog.increment();
    rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());
    func_apply(update_hyper_sgd_eta, std::move(mean_eta_tup));
    func_apply(update_hyper_sgd_eta, std::move(var_eta_tup));

    conv.add(model.llik().transpose() * onesN);
    bool converged = conv.converged(opt.vbtol(), opt.miniter());
    if (opt.verbose())
      conv.print(Rcpp::Rcerr);
    if (converged) {
      TLOG("Converged hyperparameter log-likelihood");
      break;
    }
  }

  TLOG("Finished SGVB inference");
  return conv.summarize();
}

#endif
