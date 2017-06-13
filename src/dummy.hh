#include <RcppCommon.h>

#ifndef DUMMY_EFFECT_HH_
#define DUMMY_EFFECT_HH_

struct dummy_mat_t {
  operator SEXP() const { return Rcpp::NumericVector(); }
};

struct dummy_eta_t {
  inline const dummy_mat_t &repr_mean() const { return dummy_mat; }
  template <typename T>
  inline const dummy_mat_t &sample(T &) const {
    return dummy_mat;
  }
  void resolve() const {}
  dummy_mat_t dummy_mat;
};

////////////////////////////////////////////////////////////////

template <typename Derived>
inline const Eigen::MatrixBase<Derived> &operator+(
    const Eigen::MatrixBase<Derived> &lhs, const dummy_mat_t &) {
  return lhs;
}

template <typename Derived>
inline const Eigen::MatrixBase<Derived> &operator+=(
    const Eigen::MatrixBase<Derived> &lhs, const dummy_mat_t &) {
  return lhs;
}

template <typename Derived>
inline const Eigen::MatrixBase<Derived> &operator+(
    const dummy_mat_t &, const Eigen::MatrixBase<Derived> &rhs) {
  return rhs;
}

template <typename Derived>
inline const Eigen::MatrixBase<Derived> &operator+=(
    const dummy_mat_t &, const Eigen::MatrixBase<Derived> &rhs) {
  return rhs;
}

template <typename Derived>
inline const Eigen::SparseMatrixBase<Derived> &operator+(
    const Eigen::SparseMatrixBase<Derived> &lhs, const dummy_mat_t &) {
  return lhs;
}

template <typename Derived>
inline const Eigen::SparseMatrixBase<Derived> &operator+(
    const dummy_mat_t &, const Eigen::SparseMatrixBase<Derived> &rhs) {
  return rhs;
}

template <typename T>
struct is_dummy_eta_type {
  static constexpr bool value = false;
};

template <>
struct is_dummy_eta_type<dummy_eta_t> {
  static constexpr bool value = true;
};

#endif
