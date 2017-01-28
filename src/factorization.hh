////////////////////////////////////////////////////////////////
// A wrapper for eta = U * V

#include <memory>
#include <type_traits>

#include "gaus_repr.hh"
#include "parameters.hh"
#include "eigen_util.hh"
#include "rcpp_util.hh"

#ifndef FACTORIZATION_HH_
#define FACTORIZATION_HH_

template <typename Repr, typename UParam, typename VParam>
struct factorization_t;

template <typename UParam, typename VParam, typename Scalar, typename Matrix>
struct get_factorization_type;

template <typename UParam, typename VParam, typename Scalar>
struct get_factorization_type<UParam, VParam, Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > {
  using type = factorization_t<DenseReprMat<Scalar>, UParam, VParam>;
};

template <typename UParam, typename VParam, typename Scalar>
struct get_factorization_type<UParam, VParam, Scalar, Eigen::SparseMatrix<Scalar> > {
  using type = factorization_t<SparseReprMat<Scalar>, UParam, VParam>;
};

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_eta(const Eigen::MatrixBase<Derived>& data, UParam& u, VParam& v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_t<DenseReprMat<Scalar>, UParam, VParam>;
  return Fact(data.derived(), u, v);
}

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_eta(const Eigen::SparseMatrixBase<Derived>& data, UParam& u, VParam& v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_t<SparseReprMat<Scalar>, UParam, VParam>;
  return Fact(data.derived(), u, v);
}

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_eta_ptr(const Eigen::MatrixBase<Derived>& data, UParam& u, VParam& v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_t<DenseReprMat<Scalar>, UParam, VParam>;
  return std::make_shared<Fact>(data.derived(), u, v);
}

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_eta_ptr(const Eigen::SparseMatrixBase<Derived>& data, UParam& u, VParam& v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_t<SparseReprMat<Scalar>, UParam, VParam>;
  return std::make_shared<Fact>(data.derived(), u, v);
}

template <typename Repr, typename UParam, typename VParam>
struct factorization_t {
  using UParamMat = typename param_traits<UParam>::Matrix;
  using VParamMat = typename param_traits<VParam>::Matrix;
  using DataMat = typename Repr::DataMatrix;
  using Scalar = typename DataMat::Scalar;
  using Index = typename DataMat::Index;

  explicit factorization_t(const DataMat& data, UParam& u, VParam& v)
      : Eta(make_gaus_repr(data)),
        U(u),
        V(v),
        n(U.rows()),
        m(V.rows()),
        k(U.cols()),
        nobs_u(n, k),
        nobs_v(m, k),
        u_sq(n, k),
        v_sq(m, k),
        grad_u_mean(n, k),
        grad_u_var(n, k),
        grad_v_mean(m, k),
        grad_v_var(m, k) {
    copy_matrix(mean_param(u), nobs_u);
    copy_matrix(mean_param(v), nobs_v);

    using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Dense ones_mk = Dense::Ones(m, k);
    Dense ones_nk = Dense::Ones(n, k);

    XY_nobs(data, ones_mk, nobs_u);
    XtY_nobs(data, ones_nk, nobs_v);

    copy_matrix(mean_param(u), u_sq);
    copy_matrix(mean_param(v), v_sq);
    copy_matrix(mean_param(u), grad_u_mean);
    copy_matrix(var_param(u), grad_u_var);
    copy_matrix(mean_param(v), grad_v_mean);
    copy_matrix(var_param(v), grad_v_var);

    setConstant(u_sq, 0.0);
    setConstant(v_sq, 0.0);

    resolve_param(U);
    resolve_param(V);

    resolve();
  }

  void jitter(const Scalar sd) {
    perturb_param(U, sd);
    perturb_param(V, sd);

    resolve_param(U);
    resolve_param(V);

    resolve();
  }

  Repr Eta;
  UParam& U;
  VParam& V;

  const Index n;
  const Index m;
  const Index k;

  UParamMat nobs_u;
  VParamMat nobs_v;

  UParamMat u_sq;
  VParamMat v_sq;

  UParamMat grad_u_mean;
  UParamMat grad_u_var;
  VParamMat grad_v_mean;
  VParamMat grad_v_var;

  template <typename RAND>
  const DataMat& sample(const RAND& rnorm) {
    return sample_repr(Eta, rnorm);
  }
  const DataMat& repr_mean() const { return Eta.get_mean(); }
  const DataMat& repr_var() const { return Eta.get_var(); }

  void add_sgd(const DataMat& llik) { update_repr(Eta, llik); }

  // mean = E[U] * E[V']
  // var  = Var[U] * Var[V'] + E[U]^2 * Var[V'] + Var[U] * E[V']^2
  void resolve() {
    update_mean(Eta, mean_param(U) * mean_param(V).transpose());

    update_var(Eta, var_param(U) * var_param(V).transpose() +
                        mean_param(U).unaryExpr(square_op) * var_param(V).transpose() +
                        var_param(U) * mean_param(V).unaryExpr(square_op).transpose());
  }

  ///////////////////////////////////////////////////////////////////////
  // (1) Gradient w.r.t. E[U]                                          //
  //     G1 * partial mean / partial E[U] = G1 * E[V]                  //
  //     G2 * partial var  / partial E[U] = 2 * (G2 * Var[V]) .* E[U]  //
  //                                                                   //
  // (2) Gradient w.r.t. Var[U]                                        //
  //     G2 * partial var / partial var[U] = G2 * Var[V]               //
  //                                      += G2 * E[V]^2               //
  // (3) Gradient w.r.t. E[V]                                          //
  //     G1 * partial mean / partial E[V] = G1' * E[U]                 //
  //     G2 * partial var  / partial E[V] = 2 * (G2' * Var[U]) .* E[V] //
  //                                                                   //
  // (4) Gradient w.r.t. Var[U]                                        //
  //     G2 * partial var / partial var[U] = G2' * Var[U]              //
  //                                      += G2' * E[U]^2              //
  ///////////////////////////////////////////////////////////////////////

  void _compute_sgd_u() {
    times_set(Eta.get_grad_type2(), var_param(V), grad_u_mean);
    grad_u_mean = grad_u_mean.cwiseProduct(mean_param(U)) * 2.0;
    times_add(Eta.get_grad_type1(), mean_param(V), grad_u_mean);

    times_set(Eta.get_grad_type2(), v_sq, grad_u_var);
    times_add(Eta.get_grad_type2(), var_param(V), grad_u_var);
  }

  void _compute_sgd_v() {
    trans_times_set(Eta.get_grad_type2(), var_param(U), grad_v_mean);
    grad_v_mean = grad_v_mean.cwiseProduct(mean_param(V)) * 2.0;
    trans_times_add(Eta.get_grad_type1(), mean_param(U), grad_v_mean);

    trans_times_set(Eta.get_grad_type2(), u_sq, grad_v_var);
    trans_times_add(Eta.get_grad_type2(), var_param(U), grad_v_var);
  }

  void eval_sgd() {
    Eta.summarize();

    u_sq = mean_param(U);
    u_sq = u_sq.cwiseProduct(u_sq);
    v_sq = mean_param(V);
    v_sq = v_sq.cwiseProduct(v_sq);

    this->_compute_sgd_u();  // gradient for u
    eval_param_sgd(U, grad_u_mean, grad_u_var, nobs_u);
    this->_compute_sgd_v();  // gradient for v
    eval_param_sgd(V, grad_v_mean, grad_v_var, nobs_v);
  }

  void eval_hyper_sgd() {
    Eta.summarize();

    u_sq = mean_param(U);
    u_sq = u_sq.cwiseProduct(u_sq);
    v_sq = mean_param(V);
    v_sq = v_sq.cwiseProduct(v_sq);

    this->_compute_sgd_u();  // gradient for u
    eval_hyperparam_sgd(U, grad_u_mean, grad_u_var, nobs_u);
    this->_compute_sgd_v();  // gradient for v
    eval_hyperparam_sgd(V, grad_v_mean, grad_v_var, nobs_v);
  }

  void update_sgd(const Scalar rate) {
    update_param_sgd(U, rate);
    update_param_sgd(V, rate);
    resolve_param(U);
    resolve_param(V);
    this->resolve();
  }

  void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(U, rate);
    update_hyperparam_sgd(V, rate);
    resolve_hyperparam(U);
    resolve_hyperparam(V);
    resolve_param(U);
    resolve_param(V);
    this->resolve();
  }

  struct square_op_t {
    Scalar operator()(const Scalar& x) const { return x * x; }
  } square_op;
};

#endif
