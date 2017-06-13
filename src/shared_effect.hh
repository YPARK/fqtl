#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef SHAREDEFFECT_HH_
#define SHAREDEFFECT_HH_

///////////////////////////////
// Eta ~ X * theta * t(ones) //
///////////////////////////////

template <typename Repr, typename Param>
struct sharedeffect_t;

template <typename Param, typename Scalar, typename Matrix>
struct get_sharedeffect_type;

template <typename Param, typename Scalar>
struct get_sharedeffect_type<
    Param, Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > {
  using type = sharedeffect_t<DenseReprMat<Scalar>, Param>;
};

template <typename Param, typename Scalar>
struct get_sharedeffect_type<Param, Scalar, Eigen::SparseMatrix<Scalar> > {
  using type = sharedeffect_t<SparseReprMat<Scalar>, Param>;
};

template <typename xDerived, typename yDerived, typename Param>
auto make_sharedeffect_eta(const Eigen::MatrixBase<xDerived>& xx,
                           const Eigen::MatrixBase<yDerived>& yy,
                           Param& theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = sharedeffect_t<DenseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_sharedeffect_eta(const Eigen::SparseMatrixBase<xDerived>& xx,
                           const Eigen::SparseMatrixBase<yDerived>& yy,
                           Param& theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = sharedeffect_t<SparseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_sharedeffect_eta_ptr(const Eigen::MatrixBase<xDerived>& xx,
                               const Eigen::MatrixBase<yDerived>& yy,
                               Param& theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = sharedeffect_t<DenseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_sharedeffect_eta_ptr(const Eigen::SparseMatrixBase<xDerived>& xx,
                               const Eigen::SparseMatrixBase<yDerived>& yy,
                               Param& theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = sharedeffect_t<SparseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

////////////////////////////
// Eta ~ ones * theta * X //
////////////////////////////

template <typename Repr, typename Param>
struct transpose_sharedeffect_t;

template <typename Param, typename Scalar, typename Matrix>
struct get_transpose_sharedeffect_type;

template <typename Param, typename Scalar>
struct get_transpose_sharedeffect_type<
    Param, Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > {
  using type = transpose_sharedeffect_t<DenseReprMat<Scalar>, Param>;
};

template <typename Param, typename Scalar>
struct get_transpose_sharedeffect_type<Param, Scalar,
                                       Eigen::SparseMatrix<Scalar> > {
  using type = transpose_sharedeffect_t<SparseReprMat<Scalar>, Param>;
};

template <typename xDerived, typename yDerived, typename Param>
auto make_transpose_sharedeffect_eta(const Eigen::MatrixBase<xDerived>& xx,
                                     const Eigen::MatrixBase<yDerived>& yy,
                                     Param& theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = transpose_sharedeffect_t<DenseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_transpose_sharedeffect_eta(
    const Eigen::SparseMatrixBase<xDerived>& xx,
    const Eigen::SparseMatrixBase<yDerived>& yy, Param& theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = transpose_sharedeffect_t<SparseReprMat<Scalar>, Param>;
  return Reg(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_transpose_sharedeffect_eta_ptr(const Eigen::MatrixBase<xDerived>& xx,
                                         const Eigen::MatrixBase<yDerived>& yy,
                                         Param& theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = transpose_sharedeffect_t<DenseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

template <typename xDerived, typename yDerived, typename Param>
auto make_transpose_sharedeffect_eta_ptr(
    const Eigen::SparseMatrixBase<xDerived>& xx,
    const Eigen::SparseMatrixBase<yDerived>& yy, Param& theta) {
  using Scalar = typename yDerived::Scalar;
  using Reg = transpose_sharedeffect_t<SparseReprMat<Scalar>, Param>;
  return std::make_shared<Reg>(xx.derived(), yy.derived(), theta);
}

////////////////////////////////////////////////////////////////
// A wrapper for eta = X   * theta * 1'
//                    n x p  p x m   m x 1
template <typename Repr, typename Param>
struct sharedeffect_t {
  using ParamMatrix = typename param_traits<Param>::Matrix;
  using Scalar = typename param_traits<Param>::Scalar;
  using Index = typename param_traits<Param>::Index;
  using ReprMatrix = typename Repr::DataMatrix;

  explicit sharedeffect_t(const ReprMatrix& xx, const ReprMatrix& yy,
                          Param& theta)
      : n(xx.rows()),
        p(xx.cols()),
        m(yy.cols()),
        Nobs(p, 1),
        Theta(theta),
        X(n, p),
        Xsq(n, p),
        XtG1sum(p, 1),
        X2tG2sum(p, 1),
        onesM(m, 1),
        tempN(n, 1),
        Eta(make_gaus_repr(yy)) {
    ///////////////
    // check dim //
    ///////////////

    check_dim(xx, n, p, "X in sharedeffect_t");
    check_dim(Theta, p, static_cast<Index>(1), "Theta in sharedeffect_t");
    check_dim(Eta, n, m, "Eta in sharedeffect_t");
    copy_matrix(mean_param(Theta), Nobs);
    onesM.setOnes();

    // 1. compute Nobs
    // 2. copy X and Xsq removing missing values
    // 3. create representation Eta
    XYZ_nobs(xx.transpose(), yy, onesM, Nobs);  // [p x n] [n x m] [m x 1]
    copy_matrix(Nobs, XtG1sum);
    copy_matrix(Nobs, X2tG2sum);

    remove_missing(xx, X);
    Xsq = X.cwiseProduct(X);
    resolve();
  }

  const Index n;
  const Index p;
  const Index m;

  ParamMatrix Nobs;
  Param& Theta;
  ReprMatrix X;
  ReprMatrix Xsq;
  ParamMatrix XtG1sum;
  ParamMatrix X2tG2sum;
  ParamMatrix onesM;
  ParamMatrix tempN;
  Repr Eta;

  void resolve() {
    update_mean(Eta, X * mean_param(Theta) *
                         onesM.transpose());  // [n x p] [p x 1] [1 x m]
    update_var(Eta, Xsq * var_param(Theta) *
                        onesM.transpose());  // [n x p] [p x 1] [1 x m]
  }

  template <typename RAND>
  const ReprMatrix& sample(const RAND& rnorm) {
    return sample_repr(Eta, rnorm);
  }
  const ReprMatrix& repr_mean() const { return Eta.get_mean(); }
  const ReprMatrix& repr_var() const { return Eta.get_var(); }

  void add_sgd(const ReprMatrix& llik) { update_repr(Eta, llik); }

  void eval_sgd() {
    Eta.summarize();

    tempN = Eta.get_grad_type1() * onesM;   // [n x m] [m x 1]
    trans_times_set(X, tempN, XtG1sum);     // [p x n] [n x 1]
    tempN = Eta.get_grad_type2() * onesM;   // [n x m] [m x 1]
    trans_times_set(Xsq, tempN, X2tG2sum);  // [p x n] [n x 1]
    eval_param_sgd(Theta, XtG1sum, X2tG2sum, Nobs);
  }

  void update_sgd(const Scalar rate) {
    update_param_sgd(Theta, rate);
    resolve_param(Theta);
    resolve();
  }

  void eval_hyper_sgd() {
    Eta.summarize();

    tempN = Eta.get_grad_type1() * onesM;
    trans_times_set(X, tempN, XtG1sum);
    tempN = Eta.get_grad_type2() * onesM;
    trans_times_set(Xsq, tempN, X2tG2sum);
    eval_hyperparam_sgd(Theta, XtG1sum, X2tG2sum, Nobs);
  }

  void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(Theta, rate);
    resolve_hyperparam(Theta);
    resolve();
  }
};

////////////////////////////////////////////////////////////////
// A wrapper for eta =  1  * theta * X
//                    n x 1  1 x p   p x m
template <typename Repr, typename Param>
struct transpose_sharedeffect_t {
  using ParamMatrix = typename param_traits<Param>::Matrix;
  using Scalar = typename param_traits<Param>::Scalar;
  using Index = typename param_traits<Param>::Index;
  using ReprMatrix = typename Repr::DataMatrix;

  explicit transpose_sharedeffect_t(const ReprMatrix& xx, const ReprMatrix& yy,
                                    Param& theta)
      : n(yy.rows()),
        p(xx.rows()),
        m(yy.cols()),
        Nobs(1, p),
        Theta(theta),
        X(p, m),
        Xsq(p, m),
        sumG1Xt(1, p),
        sumG2X2t(1, p),
        onesN(n, 1),
        tempM(1, m),
        Eta(make_gaus_repr(yy)) {
    check_dim(xx, p, m, "X in sharedeffect_t");
    check_dim(Theta, static_cast<Index>(1), p,
              "Theta in transpose_sharedeffect_t");
    check_dim(Eta, n, m, "Eta in transpose_sharedeffect_t");
    copy_matrix(mean_param(Theta), Nobs);
    onesN.setOnes();

    // 1. compute Nobs
    // 2. copy X and Xsq removing missing values
    // 3. create representation Eta
    XYZ_nobs(onesN.transpose(), yy, xx.transpose(), Nobs);
    copy_matrix(Nobs, sumG1Xt);
    copy_matrix(Nobs, sumG2X2t);

    remove_missing(xx, X);
    Xsq = X.cwiseProduct(X);
    resolve();
  }

  const Index n;
  const Index p;
  const Index m;

  ParamMatrix Nobs;
  Param& Theta;
  ReprMatrix X;
  ReprMatrix Xsq;
  ParamMatrix sumG1Xt;
  ParamMatrix sumG2X2t;
  ParamMatrix onesN;
  ParamMatrix tempM;
  Repr Eta;

  void resolve() {
    update_mean(Eta, onesN * mean_param(Theta) * X);  // [n x 1] [1 x p] [p x m]
    update_var(Eta, onesN * var_param(Theta) * Xsq);  // [n x 1] [1 x p] [p x m]
  }

  template <typename RAND>
  const ReprMatrix& sample(const RAND& rnorm) {
    return sample_repr(Eta, rnorm);
  }
  const ReprMatrix& repr_mean() const { return Eta.get_mean(); }
  const ReprMatrix& repr_var() const { return Eta.get_var(); }

  void add_sgd(const ReprMatrix& llik) { update_repr(Eta, llik); }

  void eval_sgd() {
    Eta.summarize();

    tempM = onesN.transpose() * Eta.get_grad_type1();  // [1 x n] [n x m]
    times_set(tempM, X.transpose(), sumG1Xt);          // [1 x m] [m x p]
    tempM = onesN.transpose() * Eta.get_grad_type2();  // [1 x n] [n x m]
    times_set(tempM, Xsq.transpose(), sumG2X2t);       // [1 x m] [m x p]
    eval_param_sgd(Theta, sumG1Xt, sumG2X2t, Nobs);
  }

  void update_sgd(const Scalar rate) {
    update_param_sgd(Theta, rate);
    resolve_param(Theta);
    resolve();
  }

  void eval_hyper_sgd() {
    Eta.summarize();

    tempM = onesN.transpose() * Eta.get_grad_type1();
    times_set(tempM, X.transpose(), sumG1Xt);
    tempM = onesN.transpose() * Eta.get_grad_type2();
    times_set(tempM, Xsq.transpose(), sumG2X2t);
    eval_hyperparam_sgd(Theta, sumG1Xt, sumG2X2t, Nobs);
  }

  void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(Theta, rate);
    resolve_hyperparam(Theta);
    resolve();
  }
};

#endif
