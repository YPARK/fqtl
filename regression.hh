////////////////////////////////////////////////////////////////
// A wrapper for eta = X * Theta -> Y
//
// (1) Theta is only created once, and referenced by many eta's.
//
// (2) Many eta's can be created to accommodate different random
// selection of data points (i.e., rows).
//

#include <memory>
#include <type_traits>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "util.hh"

#ifndef REGRESSION_HH_
#define REGRESSION_HH_

template<typename Repr, typename Param>
struct regression_t;

////////////////////////////////////////////////////////////////
template<typename REG, typename RAND_NORM>
const typename REG::ReprMatrix &sample(REG &reg, const RAND_NORM &rnorm) {
  return sample_repr(reg.Eta, rnorm);
}

template<typename REG, typename Left, typename Right, typename RAND_NORM>
const typename REG::ReprMatrix &sample(REG &reg, const Left &cholLeft, const Right &cholRight, const RAND_NORM &rnorm) {
  return sample_repr(reg.Eta, cholLeft, cholRight, rnorm);
}

////////////////////////////////////////////////////////////////
template<typename Param, typename Scalar, typename Matrix>
struct get_regression_type;

template<typename Param, typename Scalar>
struct get_regression_type<
    Param, Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
  using type = regression_t<DenseReprMat<Scalar>, Param>;
};

template<typename Param, typename Scalar>
struct get_regression_type<Param, Scalar, Eigen::SparseMatrix<Scalar>> {
  using type = regression_t<SparseReprMat<Scalar>, Param>;
};

////////////////////////////////////////////////////////////////
template<typename xDerived, typename yDerived, typename Param>
auto make_regression_eta(const Eigen::MatrixBase<xDerived> &xx,
                         const Eigen::MatrixBase<yDerived> &yy, Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = regression_t<DenseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template<typename xDerived, typename yDerived, typename Param>
auto make_regression_eta(const Eigen::SparseMatrixBase<xDerived> &xx,
                         const Eigen::SparseMatrixBase<yDerived> &yy,
                         Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = regression_t<SparseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

////////////////////////////////////////////////////////////////
template<typename xDerived, typename yDerived, typename Param>
auto make_regression_eta_ptr(const Eigen::MatrixBase<xDerived> &xx,
                             const Eigen::MatrixBase<yDerived> &yy,
                             Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = regression_t<DenseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

template<typename xDerived, typename yDerived, typename Param>
auto make_regression_eta_ptr(const Eigen::SparseMatrixBase<xDerived> &xx,
                             const Eigen::SparseMatrixBase<yDerived> &yy,
                             Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = regression_t<SparseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

////////////////////////////////////////////////////////////////
// X -> Y: regression of Y on X
// Can we share Eta?
template<typename Repr, typename Param>
struct regression_t {
  using ParamMatrix = typename param_traits<Param>::Matrix;
  using Scalar = typename param_traits<Param>::Scalar;
  using Index = typename param_traits<Param>::Index;
  using ReprMatrix = typename Repr::DataMatrix;

  explicit regression_t(const ReprMatrix &xx,
                        const ReprMatrix &yy,
                        Param &theta)
      : n(xx.rows()), p(xx.cols()), m(yy.cols()), Nobs(p, m), Theta(theta),
        X(n, p), Xsq(n, p), Xt(n, p), Xtsq(n, p), XtG1(p, m), X2tG2(p, m),
        Eta(make_gaus_repr(yy)) {
    init(xx, yy);
  }

 private:
  void init(const ReprMatrix &xx, const ReprMatrix &yy) {
#ifdef DEBUG
    check_dim(Theta, p, m, "Theta in regression_t");
    check_dim(Eta, n, m, "Eta in regression_t");
#endif
    copy_matrix(mean_param(Theta), Nobs);

    // 1. compute Nobs
    // 2. copy X and Xsq removing missing values
    // 3. create representation Eta
    XtY_nobs(xx, yy, Nobs);
    copy_matrix(Nobs, XtG1);
    copy_matrix(Nobs, X2tG2);

    remove_missing(xx, X);
    remove_missing(xx.unaryExpr([](const auto &x) { return x * x; }), Xsq);
    resolve();

    Xt = X.transpose();
    Xtsq = Xsq.transpose();
  }

 public:
  const Index n;
  const Index p;
  const Index m;

  ParamMatrix Nobs;
  Param &Theta;
  ReprMatrix X;
  ReprMatrix Xsq;
  ReprMatrix Xt;
  ReprMatrix Xtsq;
  ParamMatrix XtG1;
  ParamMatrix X2tG2;
  Repr Eta;

  void resolve() {
    update_mean(Eta, X * mean_param(Theta));
    update_var(Eta, Xsq * var_param(Theta));
  }

  const ReprMatrix &repr_mean() const { return Eta.get_mean(); }

  const ReprMatrix &repr_var() const { return Eta.get_var(); }

  void add_sgd(const ReprMatrix &llik) { update_gradient(Eta, llik); }

  void eval_sgd() {
    Eta.summarize();
    trans_times_set(X, Eta.get_grad_type1(), XtG1);
    trans_times_set(Xsq, Eta.get_grad_type2(), X2tG2);
    eval_param_sgd(Theta, XtG1, X2tG2, Nobs);
  }

  void update_sgd(const Scalar rate) {
    update_param_sgd(Theta, rate);
    resolve_param(Theta);
    resolve();
  }

  void eval_hyper_sgd() {
    Eta.summarize();
    trans_times_set(X, Eta.get_grad_type1(), XtG1);
    trans_times_set(Xsq, Eta.get_grad_type2(), X2tG2);
    eval_hyperparam_sgd(Theta, XtG1, X2tG2, Nobs);
  }

  void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(Theta, rate);
    resolve_hyperparam(Theta);
    resolve();
  }

}; // End of regression class


#endif // End of regression_hh
