#include <cmath>
#include <memory>
#include "eigen_util.hh"

#ifndef ADAM_HH_
#define ADAM_HH_

////////////////////////////////////////////////////////////////
// Gradient calculation with adaptive learning rate
// (Kingma & Ba, ICLR, 2015)
////////////////////////////////////////////////////////////////
template <typename T>
struct is_adam_scalar {
  static const bool value = false;
};

template <>
struct is_adam_scalar<float> {
  static const bool value = true;
};

template <>
struct is_adam_scalar<double> {
  static const bool value = true;
};

template <typename T, typename Scalar, typename Index>
struct adam_t;

template <typename Scalar, typename Index>
void reset_adam(adam_t<Scalar, Scalar, Index>& adam, const Scalar r_m,
                const Scalar r_v, const Index n1, const Index n2) {
  adam.G = 0;
  adam.M = 0;
  adam.V = 0;
}

template <typename Mat, typename Scalar, typename Index>
void reset_adam(adam_t<Mat, Scalar, Index>& adam, const Scalar r_m,
                const Scalar r_v, const Index n1, const Index n2) {
  adam.G.resize(n1, n2);
  adam.M.resize(n1, n2);
  adam.V.resize(n1, n2);
  setConstant(adam.G, 0.0);
  setConstant(adam.M, 0.0);
  setConstant(adam.V, 0.0);
}

template <typename Scalar>
struct adam_update_op_t {
  explicit adam_update_op_t(const Scalar& _step, const Scalar r_m,
                            const Scalar r_v)
      : step(_step), rate_m(r_m), rate_v(r_v) {}
  const Scalar operator()(const Scalar mm, const Scalar vv) const {
    const Scalar mm_stuff = mm / (one_val - std::pow(rate_m, step));
    const Scalar vv_stuff = vv / (one_val - std::pow(rate_v, step));
    const Scalar ret = mm_stuff / (std::sqrt(vv_stuff) + eps);
    return ret;
  }
  const Scalar& step;
  const Scalar rate_m;
  const Scalar rate_v;
  static constexpr Scalar eps = 1e-8;
  static constexpr Scalar one_val = 1.0;
};

template <typename T, typename Scalar, typename Index = unsigned int>
struct adam_t {
  explicit adam_t(const Scalar r_m, const Scalar r_v, const Index n1 = 1,
                  const Index n2 = 1)
      : t(0.0), update_op(this->t, r_m, r_v), rate_m(r_m), rate_v(r_v) {
    reset_adam(*this, r_m, r_v, n1, n2);
  }

  T G;
  T M;
  T V;
  Scalar t;
  adam_update_op_t<Scalar> update_op;

  const Scalar rate_m;
  const Scalar rate_v;
};

template <typename T, typename Scalar>
const T& update_adam(adam_t<T, Scalar>& adam,
                     const Eigen::MatrixBase<T>& newG) {
  adam.t++;
  adam.M = adam.M * adam.rate_m + newG * (1.0 - adam.rate_m);
  adam.V = adam.V * adam.rate_v + newG.cwiseProduct(newG) * (1.0 - adam.rate_v);
  adam.G = adam.M.binaryExpr(adam.V, adam.update_op);
  return adam.G;
}

template <typename T, typename Scalar>
const T& update_adam(adam_t<T, Scalar>& adam,
                     const Eigen::SparseMatrixBase<T>& newG) {
  adam.t++;
  adam.M = adam.M * adam.rate_m + newG * (1.0 - adam.rate_m);
  adam.V = adam.V * adam.rate_v + newG.cwiseProduct(newG) * (1.0 - adam.rate_v);
  adam.G = adam.M.binaryExpr(adam.V, adam.update_op);
  return adam.G;
}

template <typename Scalar>
const Scalar& update_adam(adam_t<Scalar, Scalar>& adam, const Scalar newG) {
  adam.t++;
  adam.M = adam.M * adam.rate_m + newG * (1.0 - adam.rate_m);
  adam.V = adam.V * adam.rate_v + newG * newG * (1.0 - adam.rate_v);
  adam.G = adam.update_op(adam.M, adam.V);
  return adam.G;
}

#endif
