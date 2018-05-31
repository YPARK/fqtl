#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef REGRESSION_HH_
#define REGRESSION_HH_

/////////////////////
// Eta ~ X * Theta //
/////////////////////

template <typename Repr, typename Param>
struct regression_t;

template <typename Param, typename Scalar, typename Matrix>
struct get_regression_type;

template <typename Param, typename Scalar>
struct get_regression_type<
    Param, Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
  using type = regression_t<DenseReprMat<Scalar>, Param>;
};

template <typename Param, typename Scalar>
struct get_regression_type<Param, Scalar, Eigen::SparseMatrix<Scalar>> {
  using type = regression_t<SparseReprMat<Scalar>, Param>;
};

template <typename xDerived, typename yDerived, typename Param>
auto make_regression_eta(const Eigen::MatrixBase<xDerived> &xx,
                         const Eigen::MatrixBase<yDerived> &yy, Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = regression_t<DenseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_regression_eta(const Eigen::SparseMatrixBase<xDerived> &xx,
                         const Eigen::SparseMatrixBase<yDerived> &yy,
                         Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = regression_t<SparseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_regression_eta_ptr(const Eigen::MatrixBase<xDerived> &xx,
                             const Eigen::MatrixBase<yDerived> &yy,
                             Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = regression_t<DenseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_regression_eta_ptr(const Eigen::SparseMatrixBase<xDerived> &xx,
                             const Eigen::SparseMatrixBase<yDerived> &yy,
                             Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = regression_t<SparseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

/////////////////////
// Eta ~ Theta * X //
/////////////////////

template <typename Repr, typename Param>
struct transpose_regression_t;

template <typename Param, typename Scalar, typename Matrix>
struct get_transpose_regression_type;

template <typename Param, typename Scalar>
struct get_transpose_regression_type<
    Param, Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
  using type = transpose_regression_t<DenseReprMat<Scalar>, Param>;
};

template <typename Param, typename Scalar>
struct get_transpose_regression_type<Param, Scalar,
                                     Eigen::SparseMatrix<Scalar>> {
  using type = transpose_regression_t<SparseReprMat<Scalar>, Param>;
};

template <typename xDerived, typename yDerived, typename Param>
auto make_transpose_regression_eta(const Eigen::MatrixBase<xDerived> &xx,
                                   const Eigen::MatrixBase<yDerived> &yy,
                                   Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = transpose_regression_t<DenseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_transpose_regression_eta(const Eigen::SparseMatrixBase<xDerived> &xx,
                                   const Eigen::SparseMatrixBase<yDerived> &yy,
                                   Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = transpose_regression_t<SparseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_transpose_regression_eta_ptr(const Eigen::MatrixBase<xDerived> &xx,
                                       const Eigen::MatrixBase<yDerived> &yy,
                                       Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = transpose_regression_t<DenseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_transpose_regression_eta_ptr(
    const Eigen::SparseMatrixBase<xDerived> &xx,
    const Eigen::SparseMatrixBase<yDerived> &yy, Param &theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = transpose_regression_t<SparseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

////////////////////////////////////////////////////////////////
// A wrapper for eta = X * Theta -> Y
//
// (1) Theta is only created once, and referenced by many eta's.
//
// (2) Many eta's can be created to accommodate different random
// selection of data points (i.e., rows).
//
template <typename Repr, typename Param>
struct regression_t {
  using ParamMatrix = typename param_traits<Param>::Matrix;
  using Scalar = typename param_traits<Param>::Scalar;
  using Index = typename param_traits<Param>::Index;
  using ReprMatrix = typename Repr::DataMatrix;

  explicit regression_t(const auto &xx, const auto &yy, Param &theta)
      : n(xx.rows()),
        p(xx.cols()),
        m(yy.cols()),
        Nobs(p, m),
        Theta(theta),
        X(n, p),
        Xsq(n, p),
        XtG1(p, m),
        X2tG2(p, m),
        Eta(make_gaus_repr(yy)) {
    check_dim(Theta, p, m, "Theta in regression_t");
    check_dim(Eta, n, m, "Eta in regression_t");
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
  }

  const Index n;
  const Index p;
  const Index m;

  ParamMatrix Nobs;
  Param &Theta;
  ReprMatrix X;
  ReprMatrix Xsq;
  ParamMatrix XtG1;
  ParamMatrix X2tG2;
  Repr Eta;

  void resolve() {
    update_mean(Eta, X * mean_param(Theta));
    update_var(Eta, Xsq * var_param(Theta));
  }

  template <typename RNG>
  inline Eigen::Ref<const ReprMatrix> sample(const RNG &rng) {
    return sample_repr(Eta, rng);
  }

  inline Eigen::Ref<const ReprMatrix> repr_mean() { return Eta.get_mean(); }
  inline Eigen::Ref<const ReprMatrix> repr_var() { return Eta.get_var(); }

  inline void init_by_dot(const ReprMatrix& yy, const Scalar sd) {
    ReprMatrix Y;
    remove_missing(yy, Y);
    Y = Y / static_cast<Scalar>(n);
    ReprMatrix XtY = X.transpose() * Y;
    Theta.beta = XtY * sd;
    resolve_param(Theta);
    resolve();
  }

  void add_sgd(const auto &llik) { update_repr(Eta, llik); }

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
};

////////////////////////////////////////////////////////////////
// A wrapper for eta = Theta * X -> Y
//                     n x p  p x m
// (1) Theta is only created once, and referenced by many eta's.
//
// (2) Many eta's can be created to accommodate different random
// selection of data points (i.e., rows).
//
template <typename Repr, typename Param>
struct transpose_regression_t {
  using ParamMatrix = typename param_traits<Param>::Matrix;
  using Scalar = typename param_traits<Param>::Scalar;
  using Index = typename param_traits<Param>::Index;
  using ReprMatrix = typename Repr::DataMatrix;

  explicit transpose_regression_t(const auto &xx, const auto &yy, Param &theta)
      : n(yy.rows()),
        p(xx.rows()),
        m(xx.cols()),
        Nobs(n, p),
        Theta(theta),
        X(p, m),
        Xsq(p, m),
        G1Xt(n, p),
        G2X2t(n, p),
        Eta(make_gaus_repr(yy)) {
    check_dim(Theta, n, p, "Theta in transpose_regression_t");
    check_dim(Eta, n, m, "Eta in transpose_regression_t");
    copy_matrix(mean_param(Theta), Nobs);

    // 1. compute Nobs
    // 2. copy X and Xsq removing missing values
    // 3. create representation Eta
    XY_nobs(yy, xx.transpose(), Nobs);
    copy_matrix(Nobs, G1Xt);
    copy_matrix(Nobs, G2X2t);

    remove_missing(xx, X);
    Xsq = X.cwiseProduct(X);
    resolve();
  }

  const Index n;
  const Index p;
  const Index m;

  ParamMatrix Nobs;
  Param &Theta;
  ReprMatrix X;
  ReprMatrix Xsq;
  ParamMatrix G1Xt;
  ParamMatrix G2X2t;
  Repr Eta;

  void resolve() {
    update_mean(Eta, mean_param(Theta) * X);
    update_var(Eta, var_param(Theta) * Xsq);
  }

  template <typename RNG>
  inline Eigen::Ref<const ReprMatrix> sample(const RNG &rng) {
    return sample_repr(Eta, rng);
  }

  inline Eigen::Ref<const ReprMatrix> repr_mean() { return Eta.get_mean(); }
  inline Eigen::Ref<const ReprMatrix> repr_var() { return Eta.get_var(); }

  inline void init_by_dot(const ReprMatrix &yy, const Scalar sd) {
    ReprMatrix Y;
    remove_missing(yy, Y);
    Y = Y / static_cast<Scalar>(m);
    ReprMatrix YXt = Y * X.transpose();
    Theta.beta = YXt * sd;
    resolve_param(Theta);
    resolve();
  }

  void add_sgd(const auto &llik) { update_repr(Eta, llik); }

  void eval_sgd() {
    Eta.summarize();
    times_set(Eta.get_grad_type1(), X.transpose(), G1Xt);
    times_set(Eta.get_grad_type2(), Xsq.transpose(), G2X2t);
    eval_param_sgd(Theta, G1Xt, G2X2t, Nobs);
  }

  void update_sgd(const Scalar rate) {
    update_param_sgd(Theta, rate);
    resolve_param(Theta);
    resolve();
  }

  void eval_hyper_sgd() {
    Eta.summarize();
    times_set(Eta.get_grad_type1(), X.transpose(), G1Xt);
    times_set(Eta.get_grad_type2(), Xsq.transpose(), G2X2t);
    eval_hyperparam_sgd(Theta, G1Xt, G2X2t, Nobs);
  }

  void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(Theta, rate);
    resolve_hyperparam(Theta);
    resolve();
  }
};

#endif
