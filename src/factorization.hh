////////////////////////////////////////////////////////////////
// A wrapper for eta = U * V

#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef FACTORIZATION_HH_
#define FACTORIZATION_HH_

template <typename Repr, typename UParam, typename VParam>
struct factorization_t;

template <typename UParam, typename VParam, typename Scalar, typename Matrix>
struct get_factorization_type;

template <typename UParam, typename VParam, typename Scalar>
struct get_factorization_type<
    UParam, VParam, Scalar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
  using type = factorization_t<DenseReprMat<Scalar>, UParam, VParam>;
};

template <typename UParam, typename VParam, typename Scalar>
struct get_factorization_type<UParam, VParam, Scalar,
                              Eigen::SparseMatrix<Scalar>> {
  using type = factorization_t<SparseReprMat<Scalar>, UParam, VParam>;
};

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_eta(const Eigen::MatrixBase<Derived> &data, UParam &u,
                            VParam &v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_t<DenseReprMat<Scalar>, UParam, VParam>;
  return Fact(data.derived(), u, v);
}

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_eta(const Eigen::SparseMatrixBase<Derived> &data,
                            UParam &u, VParam &v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_t<SparseReprMat<Scalar>, UParam, VParam>;
  return Fact(data.derived(), u, v);
}

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_eta_ptr(const Eigen::MatrixBase<Derived> &data,
                                UParam &u, VParam &v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_t<DenseReprMat<Scalar>, UParam, VParam>;
  return std::make_shared<Fact>(data.derived(), u, v);
}

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_eta_ptr(const Eigen::SparseMatrixBase<Derived> &data,
                                UParam &u, VParam &v) {
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

  explicit factorization_t(const DataMat &data, UParam &u, VParam &v)
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

  inline void jitter(const Scalar sd) {
    perturb_param(U, sd);
    perturb_param(V, sd);
    resolve_param(U);
    resolve_param(V);
    resolve();
  }

  template <typename Rng>
  inline void jitter(const Scalar sd, Rng &rng) {
    perturb_param(U, sd, rng);
    perturb_param(V, sd, rng);
    resolve_param(U);
    resolve_param(V);
    resolve();
  }

  inline void init_by_svd(const DataMat &data, const Scalar sd) {
    Eigen::JacobiSVD<DataMat> svd(data,
                                  Eigen::ComputeThinU | Eigen::ComputeThinV);
    DataMat uu = svd.matrixU() * sd;
    DataMat vv = svd.matrixV() * sd;
    vv = vv * svd.singularValues().asDiagonal();
    U.beta = uu.leftCols(k);
    V.beta = vv.leftCols(k);
    resolve_param(U);
    resolve_param(V);
    resolve();
  }

  Repr Eta;
  UParam &U;
  VParam &V;

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

  template <typename RNG>
  inline const DataMat &sample(const RNG &rng) {
    return sample_repr(Eta, rng);
  }
  const DataMat &repr_mean() const { return Eta.get_mean(); }
  const DataMat &repr_var() const { return Eta.get_var(); }

  inline void add_sgd(const DataMat &llik) { update_repr(Eta, llik); }

  // mean = E[U] * E[V']
  // var  = Var[U] * Var[V'] + E[U]^2 * Var[V'] + Var[U] * E[V']^2
  inline void resolve() {
    update_mean(Eta, U.theta * V.theta.transpose());

    u_sq = U.theta.cwiseProduct(U.theta);
    v_sq = V.theta.cwiseProduct(V.theta);

    update_var(Eta, U.theta_var * V.theta_var.transpose() +
                        u_sq * V.theta_var.transpose() +
                        U.theta_var * v_sq.transpose());
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

  inline void _compute_sgd_u() {
    grad_u_mean = Eta.get_grad_type1() * V.theta +
                  U.theta.cwiseProduct(Eta.get_grad_type2() * V.theta_var *
                                       static_cast<Scalar>(2.0));

    grad_u_var =
        Eta.get_grad_type2() * v_sq + Eta.get_grad_type2() * V.theta_var;
  }

  inline void _compute_sgd_v() {
    grad_v_mean = Eta.get_grad_type1().transpose() * U.theta +
                  V.theta.cwiseProduct(Eta.get_grad_type2().transpose() *
                                       U.theta_var * static_cast<Scalar>(2.0));

    grad_v_var = Eta.get_grad_type2().transpose() * u_sq +
                 Eta.get_grad_type2().transpose() * U.theta_var;
  }

  inline void eval_sgd() {
    Eta.summarize();

    u_sq = U.theta.cwiseProduct(U.theta);
    v_sq = V.theta.cwiseProduct(V.theta);

    this->_compute_sgd_u();  // gradient for u
    eval_param_sgd(U, grad_u_mean, grad_u_var, nobs_u);
    this->_compute_sgd_v();  // gradient for v
    eval_param_sgd(V, grad_v_mean, grad_v_var, nobs_v);
  }

  inline void eval_hyper_sgd() {
    Eta.summarize();

    u_sq = U.theta.cwiseProduct(U.theta);
    v_sq = V.theta.cwiseProduct(V.theta);

    this->_compute_sgd_u();  // gradient for u
    eval_hyperparam_sgd(U, grad_u_mean, grad_u_var, nobs_u);
    this->_compute_sgd_v();  // gradient for v
    eval_hyperparam_sgd(V, grad_v_mean, grad_v_var, nobs_v);
  }

  inline void update_sgd(const Scalar rate) {
    update_param_sgd(U, rate);
    update_param_sgd(V, rate);
    resolve_param(U);
    resolve_param(V);
    this->resolve();
  }

  inline void update_hyper_sgd(const Scalar rate) {
    update_hyperparam_sgd(U, rate);
    update_hyperparam_sgd(V, rate);
    resolve_hyperparam(U);
    resolve_hyperparam(V);
    resolve_param(U);
    resolve_param(V);
    this->resolve();
  }

  struct square_op_t {
    Scalar operator()(const Scalar &x) const { return x * x; }
  } square_op;
};

#endif
