//
// Created by Yongjin Park on 11/29/16.
//

#ifndef FQTL_DUMMY_HH
#define FQTL_DUMMY_HH

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>

////////////////////////////////////////////////////////////////
// dummy matrix class to handle empty effects
struct dummy_mat_t {};

struct dummy_eta_t {

  const dummy_mat_t &repr_mean() const { return dummy_mat; }

  void resolve() const {}

  dummy_mat_t dummy_mat;
};

// dummy sampling
template<typename R>
const dummy_mat_t &
sample(dummy_eta_t &eta, const R &) {
  return eta.dummy_mat;
}

// dummy sampling
template<typename C1, typename C2, typename R>
const dummy_mat_t &
sample(dummy_eta_t &eta, C1 &, C2 &, const R &) {
  return eta.dummy_mat;
}

// takes care of dummy addition
template<typename Derived>
const Eigen::MatrixBase<Derived> &
operator+(const Eigen::MatrixBase<Derived> &lhs, const dummy_mat_t &) {
  return lhs;
}

// takes care of dummy addition
template<typename Derived>
const Eigen::MatrixBase<Derived> &
operator+(const dummy_mat_t &, const Eigen::MatrixBase<Derived> &rhs) {
  return rhs;
}

// takes care of dummy addition
template<typename Derived>
const Eigen::SparseMatrixBase<Derived> &
operator+(const Eigen::SparseMatrixBase<Derived> &lhs, const dummy_mat_t &) {
  return lhs;
}

// takes care of dummy addition
template<typename Derived>
const Eigen::SparseMatrixBase<Derived> &
operator+(const dummy_mat_t &, const Eigen::SparseMatrixBase<Derived> &rhs) {
  return rhs;
}

#endif //FQTL_DUMMY_HH
