#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef RESIDUAL_HH_
#define RESIDUAL_HH_

template <typename Repr, typename Param>
struct residual_t;

template <typename Param, typename Scalar, typename Matrix>
struct get_residual_type;

template <typename Param, typename Scalar>
struct get_residual_type<
    Param,
    Scalar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
    using type = residual_t<DenseReprMat<Scalar>, Param>;
};

template <typename Param, typename Scalar>
struct get_residual_type<Param, Scalar, Eigen::SparseMatrix<Scalar>> {
    using type = residual_t<SparseReprMat<Scalar>, Param>;
};

template <typename yDerived, typename Param>
auto
make_residual_eta(const Eigen::MatrixBase<yDerived> &yy, Param &theta)
{
    using Scalar = typename yDerived::Scalar;
    using Reg = residual_t<DenseReprMat<Scalar>, Param>;
    return Reg(yy.derived(), theta);
}

template <typename yDerived, typename Param>
auto
make_residual_eta(const Eigen::SparseMatrixBase<yDerived> &yy, Param &theta)
{
    using Scalar = typename yDerived::Scalar;
    using Reg = residual_t<SparseReprMat<Scalar>, Param>;
    return Reg(yy.derived(), theta);
}

template <typename yDerived, typename Param>
auto
make_residual_eta_ptr(const Eigen::MatrixBase<yDerived> &yy, Param &theta)
{
    using Scalar = typename yDerived::Scalar;
    using Reg = residual_t<DenseReprMat<Scalar>, Param>;
    return std::make_shared<Reg>(yy.derived(), theta);
}

template <typename yDerived, typename Param>
auto
make_residual_eta_ptr(const Eigen::SparseMatrixBase<yDerived> &yy, Param &theta)
{
    using Scalar = typename yDerived::Scalar;
    using Reg = residual_t<SparseReprMat<Scalar>, Param>;
    return std::make_shared<Reg>(yy.derived(), theta);
}

////////////////////
// implementation //
////////////////////

template <typename Repr, typename Param>
struct residual_t {
    using ParamMatrix = typename param_traits<Param>::Matrix;
    using Scalar = typename param_traits<Param>::Scalar;
    using Index = typename param_traits<Param>::Index;
    using ReprMatrix = typename Repr::DataMatrix;

    explicit residual_t(const ReprMatrix &yy, Param &theta)
        : n(yy.rows())
        , m(yy.cols())
        , Nobs(n, m)
        , Theta(theta)
        , G1(n, m)
        , G2(n, m)
        , Eta(make_gaus_repr(yy))
    {
        check_dim(Theta, n, m, "Theta in residual_t");
        check_dim(Eta, n, m, "Eta in residual_t");
        copy_matrix(mean_param(Theta), Nobs);

        is_obs_op<ReprMatrix> is_obs;
        add_pseudo_op<ReprMatrix> add_pseudo(static_cast<Scalar>(1.0));
        Nobs = yy.unaryExpr(is_obs).unaryExpr(add_pseudo);
        resolve();
    }

    const Index n;
    const Index m;

    ParamMatrix Nobs;
    Param &Theta;
    ParamMatrix G1;
    ParamMatrix G2;
    Repr Eta;

    void resolve()
    {
        update_mean(Eta, mean_param(Theta));
        update_var(Eta, var_param(Theta));
    }

    template <typename RNG>
    inline Eigen::Ref<const ReprMatrix> sample(RNG &rng)
    {
        return sample_repr(Eta, rng);
    }

    inline Eigen::Ref<const ReprMatrix> repr_mean() { return Eta.get_mean(); }
    inline Eigen::Ref<const ReprMatrix> repr_var() { return Eta.get_var(); }

    void add_sgd(const ReprMatrix &llik) { update_repr(Eta, llik); }

    void eval_sgd()
    {
        Eta.summarize();
        G1 = Eta.get_grad_type1();
        G2 = Eta.get_grad_type2();
        eval_param_sgd(Theta, G1, G2, Nobs);
    }

    void update_sgd(const Scalar rate)
    {
        update_param_sgd(Theta, rate);
        resolve_param(Theta);
        resolve();
    }

    void eval_hyper_sgd()
    {
        Eta.summarize();
        G1 = Eta.get_grad_type1();
        G2 = Eta.get_grad_type2();
        eval_hyperparam_sgd(Theta, G1, G2, Nobs);
    }

    void update_hyper_sgd(const Scalar rate)
    {
        update_hyperparam_sgd(Theta, rate);
        resolve_hyperparam(Theta);
        resolve();
    }
};

#endif
