#include "engine/mathutil.hh"
#include "utils/eigen_util.hh"
#include "utils/param_check.hh"

#ifndef GAMMA_MODEL_HH_
#define GAMMA_MODEL_HH_

template <typename T>
struct gamma_model_t {
  using Scalar = typename T::Scalar;
  using Index = typename T::Index;
  using Data = T;

  struct gamma_shape_min_t : public check_positive_t<Scalar> {
    explicit gamma_shape_min_t(Scalar v) : check_positive_t<Scalar>(v) {}
  };

  struct gamma_shape_max_t : public check_positive_t<Scalar> {
    explicit gamma_shape_max_t(Scalar v) : check_positive_t<Scalar>(v) {}
  };

  template <typename X>
  using Dense = Eigen::MatrixBase<X>;

  template <typename X>
  using Sparse = Eigen::SparseMatrixBase<X>;

  explicit gamma_model_t(const T& y, const gamma_shape_min_t& shape_min, const gamma_shape_max_t& shape_max)
      : n(y.rows()),
        m(y.cols()),
        Y(n, m),
        lnY(n, m),
        llik_mat(n, m),
        sampled_mat(n, m),
        shape(n, m),
        scale(n, m),
        shape_op(shape_min.val, shape_max.val) {
    alloc_memory(y);
    gamma_model_t<T>::preprocess(y, lnY, Y);
  }

  const T& llik() const { return llik_mat; }
  const Index n;
  const Index m;

  template <typename Derived, typename OtherDerived>
  const T& eval(const Dense<Derived>& eta_mean, const Dense<OtherDerived>& eta_var) {
    return _eval(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& eval(const Sparse<Derived>& eta_mean, const Sparse<OtherDerived>& eta_var) {
    return _eval(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& sample(const Dense<Derived>& eta_mean, const Dense<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

  template <typename Derived, typename OtherDerived>
  const T& sample(const Sparse<Derived>& eta_mean, const Sparse<OtherDerived>& eta_var) {
    return _sample(eta_mean.derived(), eta_var.derived());
  }

 private:
  template <typename Derived>
  void alloc_memory(const Dense<Derived>& y_raw) {
    const auto n = y_raw.rows();
    const auto m = y_raw.cols();
    Y.setZero(n, m);
    lnY.setZero(n, m);
    llik_mat.setZero(n, m);
    sampled_mat.setZero(n, m);
    scale.setZero(n, m);
    shape.setZero(n, m);
  }

  template <typename Derived>
  void alloc_memory(const Sparse<Derived>& y_raw) {
    initialize(y_raw, llik_mat, 0.0);
    initialize(y_raw, sampled_mat, 0.0);
    initialize(y_raw, Y, 0.0);
    initialize(y_raw, lnY, 0.0);
    initialize(y_raw, shape, 0.0);
    initialize(y_raw, scale, 0.0);
  }

  template <typename Derived>
  static void preprocess(const Dense<Derived>& y_raw, Dense<Derived>& lnY, Dense<Derived>& Y_safe) {
    _preprocess(y_raw.derived(), lnY.derived(), Y_safe.derived());
  }

  template <typename Derived>
  static void preprocess(const Sparse<Derived>& y_raw, Sparse<Derived>& lnY, Sparse<Derived>& Y_safe) {
    _preprocess(y_raw.derived(), lnY.derived(), Y_safe.derived());
  }

  T Y;
  T lnY;
  T llik_mat;
  T sampled_mat;
  T shape;
  T scale;

  ////////////////////////////////////////////////////////////////
  // y    ~ Gamma(shape, scale)
  // llik = - shape * ln (scale) -ln Gamma(shape) + shape * ln (Y) - Y / scale
  //
  // scale = ln(1 + exp(eta_mean))
  // shape = Smax * sgm(-eta_var) + Smin
  //
  // e.g., Smin = 1 / max{y:y>0}, Smax = Var / min{y:y>0}
  //
  template <typename M1, typename M2>
  const T& _eval(const M1& eta_mean, const M2& eta_var) {
    shape = eta_var.unaryExpr(shape_op);
    scale = eta_mean.unaryExpr(scale_op);
    llik_mat = scale.binaryExpr(shape, llik_norm_op) + lnY.cwiseProduct(shape) - Y.cwiseQuotient(scale);
    return llik_mat;
  }

  template <typename M1, typename M2>
  const T& _sample(const M1& eta_mean, const M2& eta_var) {
    shape = eta_var.unaryExpr(shape_op);
    scale = eta_mean.unaryExpr(scale_op);
    auto rgamma = [this](const Scalar& shape_val, const Scalar& scale_val) { return rgamma_op(shape_val, scale_val); };
    sampled_mat = shape.binaryExpr(scale, rgamma);
    return sampled_mat;
  }

  ////////////////////////////////////////////////////////////////
  // helper functors
  struct shape_func_t {
    explicit shape_func_t(const Scalar smin, const Scalar smax) : Smin(smin), Smax(smax) {}
    Scalar operator()(const Scalar& eta_var) const { return Smax / (one_val + fasterexp(eta_var)) + Smin; }
    const Scalar Smin;
    const Scalar Smax;
  } shape_op;

  struct llik_norm_t {
    Scalar operator()(const Scalar& r, const Scalar& s) const { return -s * fasterlog(r) - fasterlgamma(s); }
  } llik_norm_op;

  struct scale_func_t {
    Scalar operator()(const Scalar& eta_mean) const { return small_val + log1pExp(eta_mean); }
    const log1pExp_op_t<Scalar> log1pExp;
  } scale_op;

  struct rgamma_op_t {
    using gamma_param_type = typename std::gamma_distribution<Scalar>::param_type;
    Scalar operator()(const Scalar& shape_val, const Scalar& scale_val) {
      return distrib(rng, gamma_param_type(shape_val, scale_val));
    }
    std::mt19937 rng;
    std::gamma_distribution<Scalar> distrib;
  } rgamma_op;

  // preprocess Y by adding a small number
  template <typename M>
  static void _preprocess(const M& y_raw, M& lnY, M& Y_safe) {
    auto is_valid = [](const Scalar& y) { return std::isfinite(y) && (y + small_val) > zero_val; };
    auto y_func = [&is_valid](const Scalar& y) { return is_valid(y) ? (y + small_val) : zero_val; };
    auto lny_func = [&is_valid](const Scalar& y) { return is_valid(y) ? fasterlog(y + small_val) : zero_val; };

    Y_safe = y_raw.unaryExpr(y_func);
    lnY = y_raw.unaryExpr(lny_func);
  }

  static constexpr Scalar one_val = 1.0;
  static constexpr Scalar small_val = 1e-8;
  static constexpr Scalar zero_val = 0.0;
};

#endif
