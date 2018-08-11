////////////////////////////////////////////////////////////////
// A wrapper for eta = U * V

#include <memory>
#include <type_traits>

#include "eigen_util.hh"
#include "gaus_repr.hh"
#include "parameters.hh"
#include "rcpp_util.hh"

#ifndef FACTORIZATION_WEIGHTED_HH_
#define FACTORIZATION_WEIGHTED_HH_

template <typename Repr, typename UParam, typename VParam>
struct factorization_weighted_t;

template <typename UParam, typename VParam, typename Scalar, typename Matrix>
struct get_factorization_weighted_type;

template <typename UParam, typename VParam, typename Scalar>
struct get_factorization_weighted_type<
    UParam, VParam, Scalar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> {
  using type = factorization_weighted_t<DenseReprMat<Scalar>, UParam, VParam>;
};

// template <typename UParam, typename VParam, typename Scalar>
// struct get_factorization_weighted_type<UParam, VParam, Scalar,
//                               Eigen::SparseMatrix<Scalar>> {
//   using type = factorization_weighted_t<SparseReprMat<Scalar>, UParam,
//   VParam>;
// };

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_weighted_eta(const Eigen::MatrixBase<Derived> &data,
                                     UParam &u, VParam &v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_weighted_t<DenseReprMat<Scalar>, UParam, VParam>;
  return Fact(data.derived(), u, v);
}

// template <typename Derived, typename UParam, typename VParam>
// auto make_factorization_weighted_eta(const Eigen::SparseMatrixBase<Derived>
// &data,
//                             UParam &u, VParam &v) {
//   using Scalar = typename Derived::Scalar;
//   using Fact = factorization_weighted_t<SparseReprMat<Scalar>, UParam,
//   VParam>; return Fact(data.derived(), u, v);
// }

template <typename Derived, typename UParam, typename VParam>
auto make_factorization_weighted_eta_ptr(const Eigen::MatrixBase<Derived> &data,
                                         UParam &u, VParam &v) {
  using Scalar = typename Derived::Scalar;
  using Fact = factorization_weighted_t<DenseReprMat<Scalar>, UParam, VParam>;
  return std::make_shared<Fact>(data.derived(), u, v);
}

// template <typename Derived, typename UParam, typename VParam>
// auto make_factorization_weighted_eta_ptr(const
// Eigen::SparseMatrixBase<Derived> &data,
//                                 UParam &u, VParam &v) {
//   using Scalar = typename Derived::Scalar;
//   using Fact = factorization_weighted_t<SparseReprMat<Scalar>, UParam,
//   VParam>; return std::make_shared<Fact>(data.derived(), u, v);
// }

template <typename Repr, typename UParam, typename VParam>
struct factorization_weighted_t {
  using UParamMat = typename param_traits<UParam>::Matrix;
  using VParamMat = typename param_traits<VParam>::Matrix;
  using DataMat = typename Repr::DataMatrix;
  using Scalar = typename DataMat::Scalar;
  using Index = typename DataMat::Index;

  explicit factorization_weighted_t(const DataMat &data, UParam &u, VParam &v)
      : Eta(make_gaus_repr(data)),
        U(u),
        V(v),
        n(U.rows()),
        m(V.rows()),
        k(U.cols()),
        nobs_u(n, k),
        nobs_v(m, k),
        uw_sq(n, k),
        v_sq(m, k),
        grad_u_mean(n, k),
        grad_u_var(n, k),
        grad_v_mean(m, k),
        grad_v_var(m, k),
        weight_nk(n, k),
        max_weight(1.0),
        mean_nk(n, k),
        var_nk(n, k) {
    copy_matrix(mean_param(u), nobs_u);
    copy_matrix(mean_param(v), nobs_v);

    using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Dense ones_mk = Dense::Ones(m, k);
    Dense ones_nk = Dense::Ones(n, k);

    XY_nobs(data, ones_mk, nobs_u);
    XtY_nobs(data, ones_nk, nobs_v);
    weight_nk.setConstant(max_weight);

    copy_matrix(mean_param(u), uw_sq);
    copy_matrix(mean_param(v), v_sq);
    copy_matrix(mean_param(u), grad_u_mean);
    copy_matrix(var_param(u), grad_u_var);
    copy_matrix(mean_param(v), grad_v_mean);
    copy_matrix(var_param(v), grad_v_var);

    setConstant(uw_sq, 0.0);
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

  template <typename Derived>
  inline void set_weight_nk(const Eigen::MatrixBase<Derived> &_weight,
                            const DataMat &data) {
    ASSERT(_weight.rows() == n && _weight.cols() == k, "invalid weight matrix");
    weight_nk = _weight.derived();
    const Scalar max_val =
        weight_nk.cwiseAbs().maxCoeff() + static_cast<Scalar>(1e-4);
    if (max_weight < max_val) max_weight = max_val;

    using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Dense ones_mk = Dense::Ones(m, k);
    Dense ones_nk = Dense::Ones(n, k);

    // nobs_u = (O[data] * O[1_mk]) .* weight
    // nobs_v = O[data'] * (O[1_nk] .* weight)
    DataMat nobs_nk(n, k);
    XY_nobs(data, ones_mk, nobs_nk);
    nobs_nk = nobs_nk.cwiseProduct(weight_nk);
    nobs_u = nobs_nk.array() + static_cast<Scalar>(1e-4);

    nobs_nk = weight_nk;
    nobs_nk = nobs_nk.array() + static_cast<Scalar>(1e-4);
    XtY_nobs(data, nobs_nk, nobs_v);
  }

  inline void init_by_svd(const DataMat &data, const Scalar sd) {
    DataMat Y0(n, m);
    UParamMat U0(n, k);
    VParamMat V0(m, k);

    // for each k : data .* w_k --> svd --> take the first
    for (Index j = 0; j < k; ++j) {
      for (Index c = 0; c < m; ++c) {
        Y0.col(c) = weight_nk.col(j).cwiseProduct(data.col(c));
      }

      Eigen::JacobiSVD<DataMat> svd(data,
                                    Eigen::ComputeThinU | Eigen::ComputeThinV);

      DataMat _uu = svd.matrixU() * sd;
      DataMat _vv = svd.matrixV() * sd;
      _vv = _vv * svd.singularValues().asDiagonal();

      U0.col(j) = _uu.col(0);
      V0.col(j) = _vv.col(0);
    }

    U.beta.setZero();
    V.beta.setZero();
    U.beta.leftCols(k) = U0.leftCols(k);
    V.beta.leftCols(k) = V0.leftCols(k);

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

  UParamMat uw_sq;
  VParamMat v_sq;

  UParamMat grad_u_mean;
  UParamMat grad_u_var;

  VParamMat grad_v_mean;
  VParamMat grad_v_var;

  DataMat weight_nk;  // n x k
  Scalar max_weight;

  UParamMat mean_nk;
  UParamMat var_nk;

  template <typename RNG>
  inline const auto &sample(const RNG &rng) {
    return sample_repr(Eta, rng);
  }

  const auto &repr_mean() { return Eta.get_mean(); }
  const auto &repr_var() { return Eta.get_var(); }

  inline void add_sgd(const DataMat &llik) { update_repr(Eta, llik); }

  // mean = (W .* E[U]) * E[V']
  // var  = (W^2 .* Var[U]) * Var[V'] +
  //        (W^2 .* E[U]^2) * Var[V'] + (W^2 .* Var[U]) * E[V']^2
  inline void resolve() {
    mean_nk = mean_param(U).cwiseProduct(weight_nk);
    update_mean(Eta, mean_nk * mean_param(V).transpose());

    var_nk = var_param(U).cwiseProduct(weight_nk).cwiseProduct(weight_nk);

    uw_sq = mean_nk.cwiseProduct(mean_nk);
    v_sq = mean_param(V).cwiseProduct(mean_param(V));

    update_var(Eta, var_nk * var_param(V).transpose() +
                        uw_sq * var_param(V).transpose() +
                        var_nk * v_sq.transpose());
  }

  // (1) Gradient w.r.t. E[U]
  // G1 * partial mean / partial E[U] = (G1 * E[V]) .* W
  // G2 * partial var  / partial E[U] = 2 * (G2 * Var[V]) .* E[U] .* W^2
  //
  // (2) Gradient w.r.t. Var[U]
  // G2 * partial var / partial var[U] = (G2 * Var[V] + G2 * E[V]^2) .* W^2
  //
  // (3) Gradient w.r.t. E[V]
  // G1 * partial mean / partial E[V] = G1' * (E[U] .* W)
  // G2 * partial var  / partial E[V] = 2 * (G2' * (Var[U] .* W^2)) .* E[V]
  //
  // (4) Gradient w.r.t. Var[V]
  // G2 * partial var / partial var[U] = G2' * (Var[U] .* W^2) + G2' * (E[U] .*
  // W)^2

  inline void _compute_sgd_u() {
    grad_u_mean =
        (Eta.get_grad_type1() * mean_param(V)).cwiseProduct(weight_nk) +
        (Eta.get_grad_type2() * var_param(V))
                .cwiseProduct(weight_nk)
                .cwiseProduct(weight_nk)
                .cwiseProduct(mean_param(U)) *
            static_cast<Scalar>(2.0);

    grad_u_var =
        Eta.get_grad_type2() * v_sq + Eta.get_grad_type2() * var_param(V);
    grad_u_var = grad_u_var.cwiseProduct(weight_nk).cwiseProduct(weight_nk);
  }

  inline void _compute_sgd_v() {
    grad_v_mean =
        Eta.get_grad_type1().transpose() *
            (mean_param(U).cwiseProduct(weight_nk)) +
        mean_param(V).cwiseProduct(
            Eta.get_grad_type2().transpose() *
            var_param(U).cwiseProduct(weight_nk).cwiseProduct(weight_nk)) *
            static_cast<Scalar>(2.0);

    grad_v_var =
        Eta.get_grad_type2().transpose() * uw_sq +
        Eta.get_grad_type2().transpose() *
            (var_param(U).cwiseProduct(weight_nk).cwiseProduct(weight_nk));
  }

  inline void eval_sgd() {
    Eta.summarize();

    uw_sq = mean_param(U)
                .cwiseProduct(mean_param(U))
                .cwiseProduct(weight_nk)
                .cwiseProduct(weight_nk);
    v_sq = mean_param(V).cwiseProduct(mean_param(V));

    this->_compute_sgd_u();  // gradient for u
    eval_param_sgd(U, grad_u_mean, grad_u_var, nobs_u);
    this->_compute_sgd_v();  // gradient for v
    eval_param_sgd(V, grad_v_mean, grad_v_var, nobs_v);
  }

  inline void eval_hyper_sgd() {
    Eta.summarize();

    uw_sq = mean_param(U)
                .cwiseProduct(mean_param(U))
                .cwiseProduct(weight_nk)
                .cwiseProduct(weight_nk);
    v_sq = mean_param(V).cwiseProduct(mean_param(V));

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
