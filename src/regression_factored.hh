////////////////////////////////////////////////////////////////
// A wrapper for eta = X * ThetaL * ThetaR' in Y ~ f(eta)
//
// (1) Theta is only created once, and referenced by many eta's.
// (2) Many eta's can be created to accommodate different random
// selection of data points (i.e., rows).
//

#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef FACTORED_REGRESSION_HH_
#define FACTORED_REGRESSION_HH_

template <typename Repr, typename ParamLeft, typename ParamRight>
struct factored_regression_t;

////////////////////////////////////////////////////////////////
// X -> Y: regression of Y on X
// X * ThetaL * ThetaR'
template <typename Repr, typename ParamLeft, typename ParamRight>
struct factored_regression_t {
  using ParamLeftMatrix = typename param_traits<ParamLeft>::Matrix;
  using ParamRightMatrix = typename param_traits<ParamRight>::Matrix;
  using Scalar = typename param_traits<ParamLeft>::Scalar;
  using Index = typename param_traits<ParamLeft>::Index;
  using ReprMatrix = typename Repr::DataMatrix;

  explicit factored_regression_t(const ReprMatrix &xx, const ReprMatrix &yy,
                                 ParamLeft &thetaL, ParamRight &thetaR)
      : n(xx.rows()), p(xx.cols()), m(yy.cols()), k(thetaL.cols()), NobsL(p, k),
        NobsR(m, k), ThetaL(thetaL), ThetaR(thetaR), thetaLsq(p, k),
        thetaRsq(m, k), X(n, p), Xsq(n, p), G1L(p, k), G2L(p, k), G1R(m, k),
        G2R(m, k), temp_nk(n, k), Eta(make_gaus_repr(yy)) {
#ifdef DEBUG
    check_dim(ThetaL, p, k, "ThetaL in factored_regression_t");
    check_dim(ThetaR, m, k, "ThetaR in factored_regression_t");
    check_dim(Eta, n, m, "Eta in factored_regression_t");
#endif
    copy_matrix(mean_param(ThetaL), NobsL);
    copy_matrix(mean_param(ThetaR), NobsR);

    // 1. compute Nobs
    // NobsL = O[X'] * O[Y] * O[R] (p x k)
    // NobsR = O[Y'] * O[X] * O[L] (m x k)
    XYZ_nobs(xx.transpose(), yy, mean_param(ThetaR), NobsL);
    XYZ_nobs(yy.transpose(), xx, mean_param(ThetaL), NobsR);

    // 2. copy X and Xsq removing missing values
    remove_missing(xx, X);
    remove_missing(xx.unaryExpr([](const auto &x) { return x * x; }), Xsq);

    // 3. create representation Eta
    copy_matrix(NobsL, G1L);
    copy_matrix(NobsL, G2L);
    copy_matrix(NobsR, G1R);
    copy_matrix(NobsR, G2R);

    copy_matrix(NobsL, thetaLsq);
    copy_matrix(NobsR, thetaRsq);

    setConstant(thetaLsq, 0.0);
    setConstant(thetaRsq, 0.0);

    this->resolve();
  }

  const Index n;
  const Index p;
  const Index m;
  const Index k;

  ParamLeftMatrix NobsL;  // p x k
  ParamRightMatrix NobsR; // m x k
  ParamLeft &ThetaL;      // p x k
  ParamRight &ThetaR;     // m x k

  ParamLeftMatrix thetaLsq;  // p x k
  ParamRightMatrix thetaRsq; // m x k

  ReprMatrix X;         // n x p
  ReprMatrix Xsq;       // n x p
  ParamLeftMatrix G1L;  // p x k
  ParamLeftMatrix G2L;  // p x k
  ParamRightMatrix G1R; // m x k
  ParamRightMatrix G2R; // m x k
  ReprMatrix temp_nk;   // n x k
  Repr Eta;             // n x m

  template <typename RNG> inline const ReprMatrix &sample(RNG &rng) {
    return sample_repr(Eta, rng);
  }

  const ReprMatrix &repr_mean() const { return Eta.get_mean(); }
  const ReprMatrix &repr_var() const { return Eta.get_var(); }

  inline void add_sgd(const ReprMatrix &llik) { update_repr(Eta, llik); }

  // mean = X * E[L] * E[R]'
  // var = X^2 * (Var[L] * Var[R]' + E[L]^2 * Var[R]' + Var[L] * E[R']^2)
  inline void resolve() {
    thetaRsq =
        mean_param(ThetaR).cwiseProduct(mean_param(ThetaR)); /* (m x k)  */
    thetaLsq =
        mean_param(ThetaL).cwiseProduct(mean_param(ThetaL)); /* (n x k)  */

    update_mean(Eta, X * mean_param(ThetaL) *
                         mean_param(ThetaR).transpose()); /* mean x mean */
    update_var(
        Eta,
        Xsq *
            (var_param(ThetaL) * var_param(ThetaR).transpose() + /* var x var */
             thetaLsq * var_param(ThetaR).transpose() +  /* mean^2 x var */
             var_param(ThetaL) * thetaRsq.transpose())); /* var x mean^2 */
  }

  /////////////////////////////////////////////////////////////////////////////
  // (1) gradient w.r.t. E[L]                                                //
  //     X' * G1 * E[R]                   (p x n) (n x m) (m x k)            //
  //     + 2 * X^2' * G2 * Var[R] .* E[L] (p x n) (n x m) (m x k) .* (p x k) //
  //                                                                         //
  // (2) gradient w.r.t. V[L]                                                //
  //     X^2' * G2 * (Var[R] + E[R]^2)    (p x n) (n x m) (m x k)            //
  //                                                                         //
  // (3) gradient w.r.t. E[R]                                                //
  //     G1' * X * E[L]                   (m x n) (n x p) (p x k)            //
  //     + 2 * G2' * X^2 * Var[L] .* E[R] (m x n) (n x p) (p x k) .* (m x k) //
  //                                                                         //
  // (4) gradient w.r.t. V[R]                                                //
  //     G2' * X^2 * (Var[L] + E[L]^2)    (m x n) (n x p) (p x k)            //
  /////////////////////////////////////////////////////////////////////////////

  inline void eval_sgd() {
    Eta.summarize();

    thetaRsq =
        mean_param(ThetaR).cwiseProduct(mean_param(ThetaR)); /* (m x k) */
    thetaLsq =
        mean_param(ThetaL).cwiseProduct(mean_param(ThetaL)); /* (n x k) */

    // (1) update of G1L -- reducing to [n x k] helps performance
    times_set(Eta.get_grad_type2(), var_param(ThetaR),
              temp_nk);                 /* (n x m) (m x k) = (n x k)  */
    trans_times_set(Xsq, temp_nk, G1L); /* (n x p)' (n x k) = (p x k) */
    G1L = 2.0 * G1L.cwiseProduct(mean_param(ThetaL)); /* (p x k) */

    times_set(Eta.get_grad_type1(), mean_param(ThetaR),
              temp_nk); /* (n x m) (m x k) = (n x k) */
    trans_times_add(X, temp_nk, G1L);

    // (2) update of G2L
    times_set(Eta.get_grad_type2(), var_param(ThetaR),
              temp_nk); /* (n x m) (m x k) = (n x k)  */
    times_add(Eta.get_grad_type2(), thetaRsq,
              temp_nk);                 /* (n x m) (m x k) = (n x k)  */
    trans_times_set(Xsq, temp_nk, G2L); /* (n x p)' (n x k) = (p x k) */

    eval_param_sgd(ThetaL, G1L, G2L, NobsL);

    // (3) update of G1R
    times_set(Xsq, var_param(ThetaL), temp_nk); /* (n x p) (p x k) = (n x k) */
    trans_times_set(Eta.get_grad_type2(), temp_nk,
                    G1R); /* (n x m)' (n x k) = (m x k) */
    G1R = 2.0 * G1R.cwiseProduct(mean_param(ThetaR)); /* (m x k) */

    times_set(X, mean_param(ThetaL), temp_nk); /* (n x p) (p x k) = (n x k) */
    trans_times_add(Eta.get_grad_type1(), temp_nk,
                    G1R); /* (n x m)' (n x k) = (m x k) */

    // (4) update of G2R
    times_set(Xsq, var_param(ThetaL), temp_nk); /* (n x p) (p x k) = (n x k) */
    times_add(Xsq, thetaLsq, temp_nk);          /* (n x p) (p x k) = (n x k) */
    trans_times_set(Eta.get_grad_type2(), temp_nk,
                    G2R); /* (n x m)' (n x k) = (m x k) */

    eval_param_sgd(ThetaR, G1R, G2R, NobsR);
  }

  template <typename RNG> inline void jitter(const Scalar sd, RNG &rng) {
    perturb_param(ThetaL, sd, rng);
    perturb_param(ThetaR, sd, rng);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  inline void update_sgd(const Scalar rate) {
    update_param_sgd(ThetaL, rate);
    update_param_sgd(ThetaR, rate);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  inline void eval_hyper_sgd() {
    this->eval_sgd();
    eval_hyperparam_sgd(ThetaL, G1L, G2L, NobsL);
    eval_hyperparam_sgd(ThetaR, G1R, G2R, NobsR);
  }

  inline void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(ThetaL, rate);
    update_hyperparam_sgd(ThetaR, rate);
    resolve_param(ThetaL);
    resolve_param(ThetaR);
    this->resolve();
  }

  struct square_op_t {
    Scalar operator()(const Scalar &x) const { return x * x; }
  } square_op;
};

template <typename ParamLeft, typename ParamRight, typename Scalar,
          typename Matrix>
struct get_factored_regression_type;

template <typename ParamLeft, typename ParamRight, typename Scalar>
struct get_factored_regression_type<
    ParamLeft, ParamRight, Scalar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
  using type =
      factored_regression_t<DenseReprMat<Scalar>, ParamLeft, ParamRight>;
};

template <typename ParamLeft, typename ParamRight, typename Scalar>
struct get_factored_regression_type<ParamLeft, ParamRight, Scalar,
                                    Eigen::SparseMatrix<Scalar>> {
  using type =
      factored_regression_t<SparseReprMat<Scalar>, ParamLeft, ParamRight>;
};

template <typename xDerived, typename yDerived, typename ParamLeft,
          typename ParamRight>
auto make_factored_regression_eta(const Eigen::MatrixBase<xDerived> &xx,
                                  const Eigen::MatrixBase<yDerived> &yy,
                                  ParamLeft &thetaL, ParamRight &thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg =
      factored_regression_t<DenseReprMat<Scalar>, ParamLeft, ParamRight>;
  return Reg(xx.derived(), yy.derived(), thetaL, thetaR);
}

template <typename xDerived, typename yDerived, typename ParamLeft,
          typename ParamRight>
auto make_factored_regression_eta(const Eigen::SparseMatrixBase<xDerived> &xx,
                                  const Eigen::SparseMatrixBase<yDerived> &yy,
                                  ParamLeft &thetaL, ParamRight &thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg =
      factored_regression_t<SparseReprMat<Scalar>, ParamLeft, ParamRight>;
  return Reg(xx.derived(), yy.derived(), thetaL, thetaR);
}

template <typename xDerived, typename yDerived, typename ParamLeft,
          typename ParamRight>
auto make_factored_regression_eta_ptr(const Eigen::MatrixBase<xDerived> &xx,
                                      const Eigen::MatrixBase<yDerived> &yy,
                                      ParamLeft &thetaL, ParamRight &thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg =
      factored_regression_t<DenseReprMat<Scalar>, ParamLeft, ParamRight>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), thetaL, thetaR);
}

template <typename xDerived, typename yDerived, typename ParamLeft,
          typename ParamRight>
auto make_factored_regression_eta_ptr(
    const Eigen::SparseMatrixBase<xDerived> &xx,
    const Eigen::SparseMatrixBase<yDerived> &yy, ParamLeft &thetaL,
    ParamRight &thetaR) {
  using Scalar = typename yDerived::Scalar;
  using Reg =
      factored_regression_t<SparseReprMat<Scalar>, ParamLeft, ParamRight>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), thetaL, thetaR);
}

#endif
