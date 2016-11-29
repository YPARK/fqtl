#ifndef FQTL_HH_
#define FQTL_HH_

#include <iostream>
#include <sstream>
#include <string>

#include "eigen_util.hh"
#include "factorization.hh"
#include "gaus_repr.hh"
#include "io.hh"
#include "parameters.hh"
#include "progress.hh"
#include "regression.hh"
#include "util.hh"
#include "dummy.hh"

////////////////////////////////////////////////////////////////
// update effects
template<typename Model, typename Eta, typename FixedEta,
         typename CKin, typename CTis,
         typename RNorm, typename Index, typename Scalar>
void update_eta(Model &model,
                Eta &mean_eta,
                const FixedEta &cov_eta,
                const CKin &cKin,
                const CTis &cTis,
                const RNorm &rnorm,
                const Index S,
                const Scalar rate) {
  for (auto s = 0; s < S; ++s) {
    model.eval(sample(mean_eta, cKin, cTis, rnorm) + cov_eta);
    mean_eta.add_sgd(model.llik());
  }
  mean_eta.eval_sgd();
  mean_eta.update_sgd(rate);
}

////////////////////////////////////////////////////////////////
// Handle dummy eta's
template<typename Model, typename FixedEta,
         typename CKin, typename CTis,
         typename RNorm, typename Index, typename Scalar>
void update_eta(Model &model,
                dummy_eta_t &,
                const FixedEta &cov_eta,
                const CKin &cKin,
                const CTis &cTis,
                const RNorm &rnorm,
                const Index S,
                const Scalar rate) {
  // do nothing
}

#endif // FQTL_HH_
