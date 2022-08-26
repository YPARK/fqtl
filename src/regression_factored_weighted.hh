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

#ifndef FACTORED_WEIGHTED_REGRESSION_HH_
#define FACTORED_WEIGHTED_REGRESSION_HH_

template <typename Repr, typename ParamLeft, typename ParamRight>
struct factored_weighted_regression_t;

////////////////////////////////////////////////////////////////
// X -> Y: regression of Y on X
// X * ThetaL * ThetaR'
template <typename Repr, typename ParamLeft, typename ParamRight>
struct factored_weighted_regression_t {
    using ParamLeftMatrix = typename param_traits<ParamLeft>::Matrix;
    using ParamRightMatrix = typename param_traits<ParamRight>::Matrix;
    using Scalar = typename param_traits<ParamLeft>::Scalar;
    using Index = typename param_traits<ParamLeft>::Index;
    using ReprMatrix = typename Repr::DataMatrix;

    explicit factored_weighted_regression_t(const ReprMatrix &xx,
                                            const ReprMatrix &yy,
                                            ParamLeft &thetaL,
                                            ParamRight &thetaR)
        : n(xx.rows())
        , p(xx.cols())
        , m(yy.cols())
        , k(thetaL.cols())
        , NobsL(p, k)
        , NobsR(m, k)
        , ThetaL(thetaL)
        , ThetaR(thetaR)
        , thetaLsq(p, k)
        , thetaRsq(m, k)
        , X(n, p)
        , Xsq(n, p)
        , G1L(p, k)
        , G2L(p, k)
        , G1R(m, k)
        , G2R(m, k)
        , temp_nk(n, k)
        , var_nk(n, k)
        , mean_nk(n, k)
        , weight_nk(n, k)
        , max_weight(1.0)
        , Eta(make_gaus_repr(yy))
    {
#ifdef DEBUG
        check_dim(ThetaL, p, k, "ThetaL in factored_weighted_regression_t");
        check_dim(ThetaR, m, k, "ThetaR in factored_weighted_regression_t");
        check_dim(Eta, n, m, "Eta in factored_weighted_regression_t");
#endif
        copy_matrix(ThetaL.theta, NobsL);
        copy_matrix(ThetaR.theta, NobsR);
        weight_nk.setConstant(max_weight);

        // 1. compute Nobs
        // Need to take into account of weights
        XYZ_nobs(xx.transpose(), yy, ThetaR.theta, NobsL);
        XYZ_nobs(yy.transpose(), xx, ThetaL.theta, NobsR);

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
    ReprMatrix var_nk;    // n x k
    ReprMatrix mean_nk;   // n x k

    ReprMatrix weight_nk; // n x k
    Scalar max_weight;

    Repr Eta; // n x m

    template <typename RNG>
    inline Eigen::Ref<const ReprMatrix> sample(RNG &rng)
    {
        return sample_repr(Eta, rng);
    }

    inline Eigen::Ref<const ReprMatrix> repr_mean() { return Eta.get_mean(); }
    inline Eigen::Ref<const ReprMatrix> repr_var() { return Eta.get_var(); }

    inline void add_sgd(const ReprMatrix &llik) { update_repr(Eta, llik); }

    template <typename Derived1, typename Derived2, typename Derived3>
    void set_weight_nk(const Eigen::MatrixBase<Derived1> &_weight,
                       const Eigen::MatrixBase<Derived2> &xx,
                       const Eigen::MatrixBase<Derived3> &yy)
    {
        ASSERT(_weight.rows() == n && _weight.cols() == k,
               "invalid weight matrix");

        temp_nk = _weight.derived(); // may contain nan
        remove_missing(temp_nk, weight_nk);

        const Scalar max_val =
            weight_nk.cwiseAbs().maxCoeff() + static_cast<Scalar>(1e-4);

        if (max_weight < max_val)
            max_weight = max_val;

        // NobsL = O[X'] * (weight .* (O[Y] * O[R])) -> (p x k)
        // NobsR = O[Y'] * (weight .* (O[X] * O[L])) -> (m x k)

        is_obs_op<ReprMatrix> op;
        ReprMatrix nobs_nk(n, k);

        XY_nobs(yy, ThetaR.theta, nobs_nk);
        nobs_nk = nobs_nk.cwiseProduct(weight_nk.unaryExpr(op));
        nobs_nk = nobs_nk.array() + static_cast<Scalar>(1e-4);
        XtY_nobs(xx, nobs_nk, NobsL);

        XY_nobs(xx, ThetaL.theta, nobs_nk);
        nobs_nk = nobs_nk.cwiseProduct(weight_nk.unaryExpr(op));
        nobs_nk = nobs_nk.array() + static_cast<Scalar>(1e-4);
        XtY_nobs(yy, nobs_nk, NobsR);
    }

    // mean = X * E[L] * E[R]'
    // var = X^2 * (Var[L] * Var[R]' + E[L]^2 * Var[R]' + Var[L] * E[R']^2)
    inline void resolve()
    {
        thetaRsq = ThetaR.theta.cwiseProduct(ThetaR.theta); /* (m x k)  */
        thetaLsq = ThetaL.theta.cwiseProduct(ThetaL.theta); /* (p x k)  */

        update_mean(Eta,
                    (X * ThetaL.theta).cwiseProduct(weight_nk) *
                        ThetaR.theta.transpose()); /* mean x mean */

        // mean_left_tot = X * E[L] .* weight_nk
        mean_nk = (X * ThetaL.theta).cwiseProduct(weight_nk);
        // var_left_tot = X^2 * V[L] .* (weight_nk^2)
        var_nk = (Xsq * ThetaL.theta_var)
                     .cwiseProduct(weight_nk)
                     .cwiseProduct(weight_nk);

        // var_tot = var_left_tot * V[R'] +
        //           (mean_left_tot^2) * V[R'] +
        //           var_left_tot * (E[R']^2)
        update_var(Eta,
                   var_nk * ThetaR.theta_var.transpose() + /* var x var */
                       var_nk * thetaRsq.transpose() +     /* var x mean^2 */
                       mean_nk.cwiseProduct(mean_nk) *
                           ThetaR.theta_var.transpose()); /* mean^2 x var */
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // (1) gradient w.r.t. E[L]
    //     X' * ((G1 * E[R]) .* weight_nk)
    //     + 2 * X^2' * ((G2 * Var[R]) .* weight_nk^2) .* E[L]
    //
    // (2) gradient w.r.t. V[L]
    //     X^2' * ((G2 * (Var[R] + E[R]^2)). weight_nk^2)
    //
    // (3) gradient w.r.t. E[R]
    //     G1' * ((X * E[L]) .* weight_nk)
    //     + 2 * G2' * ((X^2 * Var[L]) .* weight_nk^2) .* E[R]
    //
    // (4) gradient w.r.t. V[R]
    //     G2' * ((X^2 * (Var[L] + E[L]^2)) .* weight_nk^2)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    inline void eval_sgd()
    {
        Eta.summarize();

        thetaRsq = ThetaR.theta.cwiseProduct(ThetaR.theta); /* (m x k) */
        thetaLsq = ThetaL.theta.cwiseProduct(ThetaL.theta); /* (n x k) */

        // (1) update of G1L -- reducing to [n x k] helps performance
        // (a) on variance
        times_set(Eta.get_grad_type2(),
                  ThetaR.theta_var,
                  temp_nk); /* (n x m) (m x k) = (n x k)  */
        trans_times_set(Xsq,
                        temp_nk.cwiseProduct(weight_nk).cwiseProduct(weight_nk),
                        G1L); /* (n x p)' (n x k) = (p x k) */
        G1L = 2.0 * G1L.cwiseProduct(ThetaL.theta); /* (p x k) */

        // (b) on mean
        times_set(Eta.get_grad_type1(),
                  ThetaR.theta,
                  temp_nk); /* (n x m) (m x k) = (n x k) */
        trans_times_add(X, temp_nk.cwiseProduct(weight_nk), G1L);

        // (2) update of G2L
        times_set(Eta.get_grad_type2(),
                  ThetaR.theta_var,
                  temp_nk); /* (n x m) (m x k) = (n x k)  */
        times_add(Eta.get_grad_type2(),
                  thetaRsq,
                  temp_nk); /* (n x m) (m x k) = (n x k)  */
        trans_times_set(Xsq,
                        temp_nk.cwiseProduct(weight_nk).cwiseProduct(weight_nk),
                        G2L); /* (n x p)' (n x k) = (p x k) */

        eval_param_sgd(ThetaL, G1L, G2L, NobsL);

        // (3) update of G1R
        // (a) on varinace
        times_set(Xsq,
                  ThetaL.theta_var,
                  temp_nk); /* (n x p) (p x k) = (n x k) */
        trans_times_set(Eta.get_grad_type2(),
                        temp_nk.cwiseProduct(weight_nk).cwiseProduct(weight_nk),
                        G1R); /* (n x m)' (n x k) = (m x k) */
        G1R = 2.0 * G1R.cwiseProduct(ThetaR.theta); /* (m x k) */

        // (b) on mean
        times_set(X, ThetaL.theta, temp_nk); /* (n x p) (p x k) = (n x k) */
        trans_times_add(Eta.get_grad_type1(),
                        temp_nk.cwiseProduct(weight_nk),
                        G1R); /* (n x m)' (n x k) = (m x k) */

        // (4) update of G2R
        times_set(Xsq,
                  ThetaL.theta_var,
                  temp_nk);                /* (n x p) (p x k) = (n x k) */
        times_add(Xsq, thetaLsq, temp_nk); /* (n x p) (p x k) = (n x k) */
        trans_times_set(Eta.get_grad_type2(),
                        temp_nk.cwiseProduct(weight_nk).cwiseProduct(weight_nk),
                        G2R); /* (n x m)' (n x k) = (m x k) */

        eval_param_sgd(ThetaR, G1R, G2R, NobsR);
    }

    template <typename RNG>
    inline void jitter(const Scalar sd, RNG &rng)
    {
        perturb_param(ThetaL, sd, rng);
        perturb_param(ThetaR, sd, rng);
        resolve_param(ThetaL);
        resolve_param(ThetaR);
        this->resolve();
    }

    inline void init_by_svd(const ReprMatrix &yy, const Scalar sd)
    {
        ReprMatrix Yin;
        remove_missing(yy, Yin);

        ReprMatrix Ymean = Yin * ReprMatrix::Ones(Yin.cols(), 1) /
            static_cast<Scalar>(Yin.cols());

        ReprMatrix Y(Yin.rows(), k);
        for (Index j = 0; j < k; ++j) {
            Y.col(j) = Ymean.cwiseProduct(weight_nk.col(j));
        }

        ReprMatrix XtY = X.transpose() * Y / static_cast<Scalar>(n);
        Eigen::JacobiSVD<ReprMatrix> svd(XtY,
                                         Eigen::ComputeThinU |
                                             Eigen::ComputeThinV);
        ParamLeftMatrix left = svd.matrixU() * sd;
        ThetaL.beta.setZero();
        ThetaL.beta.leftCols(k) = left.leftCols(k);
        ThetaR.beta.setConstant(sd);
        resolve_param(ThetaL);
        resolve_param(ThetaR);
        this->resolve();
    }

    inline void update_sgd(const Scalar rate)
    {
        update_param_sgd(ThetaL, rate);
        update_param_sgd(ThetaR, rate);
        resolve_param(ThetaL);
        resolve_param(ThetaR);
        this->resolve();
    }

    inline void eval_hyper_sgd()
    {
        this->eval_sgd();
        eval_hyperparam_sgd(ThetaL, G1L, G2L, NobsL);
        eval_hyperparam_sgd(ThetaR, G1R, G2R, NobsR);
    }

    inline void update_hyper_sgd(const Scalar rate)
    {
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

template <typename ParamLeft,
          typename ParamRight,
          typename Scalar,
          typename Matrix>
struct get_factored_weighted_regression_type;

template <typename ParamLeft, typename ParamRight, typename Scalar>
struct get_factored_weighted_regression_type<
    ParamLeft,
    ParamRight,
    Scalar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
    using type = factored_weighted_regression_t<DenseReprMat<Scalar>,
                                                ParamLeft,
                                                ParamRight>;
};

// template <typename ParamLeft, typename ParamRight, typename Scalar>
// struct get_factored_weighted_regression_type<ParamLeft, ParamRight, Scalar,
//                                              Eigen::SparseMatrix<Scalar>> {
//   using type = factored_weighted_regression_t<SparseReprMat<Scalar>,
//   ParamLeft,
//                                               ParamRight>;
// };

template <typename xDerived,
          typename yDerived,
          typename ParamLeft,
          typename ParamRight>
auto
make_factored_weighted_regression_eta(const Eigen::MatrixBase<xDerived> &xx,
                                      const Eigen::MatrixBase<yDerived> &yy,
                                      ParamLeft &thetaL,
                                      ParamRight &thetaR)
{
    using Scalar = typename yDerived::Scalar;
    using Reg = factored_weighted_regression_t<DenseReprMat<Scalar>,
                                               ParamLeft,
                                               ParamRight>;
    return Reg(xx.derived(), yy.derived(), thetaL, thetaR);
}

// template <typename xDerived, typename yDerived, typename ParamLeft,
//           typename ParamRight>
// auto make_factored_weighted_regression_eta(
//     const Eigen::SparseMatrixBase<xDerived> &xx,
//     const Eigen::SparseMatrixBase<yDerived> &yy, ParamLeft &thetaL,
//     ParamRight &thetaR) {
//   using Scalar = typename yDerived::Scalar;
//   using Reg = factored_weighted_regression_t<SparseReprMat<Scalar>,
//   ParamLeft,
//                                              ParamRight>;
//   return Reg(xx.derived(), yy.derived(), thetaL, thetaR);
// }

template <typename xDerived,
          typename yDerived,
          typename ParamLeft,
          typename ParamRight>
auto
make_factored_weighted_regression_eta_ptr(const Eigen::MatrixBase<xDerived> &xx,
                                          const Eigen::MatrixBase<yDerived> &yy,
                                          ParamLeft &thetaL,
                                          ParamRight &thetaR)
{
    using Scalar = typename yDerived::Scalar;
    using Reg = factored_weighted_regression_t<DenseReprMat<Scalar>,
                                               ParamLeft,
                                               ParamRight>;
    return std::make_shared<Reg>(xx.derived(), yy.derived(), thetaL, thetaR);
}

// template <typename xDerived, typename yDerived, typename ParamLeft,
//           typename ParamRight>
// auto make_factored_weighted_regression_eta_ptr(
//     const Eigen::SparseMatrixBase<xDerived> &xx,
//     const Eigen::SparseMatrixBase<yDerived> &yy, ParamLeft &thetaL,
//     ParamRight &thetaR) {
//   using Scalar = typename yDerived::Scalar;
//   using Reg = factored_weighted_regression_t<SparseReprMat<Scalar>,
//   ParamLeft,
//                                              ParamRight>;
//   return std::make_shared<Reg>(xx.derived(), yy.derived(), thetaL, thetaR);
// }

#endif
