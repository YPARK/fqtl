#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "eigen_util.hh"
#include "param_check.hh"
#include "util.hh"
#include <fstream>
#include <iomanip>
#include <iostream>

#ifndef PROGRESS_HH_
#define PROGRESS_HH_

// Keep track of progresses of row-wise sum.
template <typename Scalar>
struct progress_t {
  using mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Index = typename mat_t::Index;

  struct Nmodels : public check_positive_t<int> {
    explicit Nmodels(const int v) : check_positive_t<int>(v) {}
  };

  struct Interv : public check_positive_t<int> {
    explicit Interv(const int v) : check_positive_t<int>(v) {}
  };

  explicit progress_t(const Nmodels& _nmodels, const Interv& _interv)
      : t(0), nrec(0), nmodels(_nmodels.val), interv(_interv.val), curr_score_stat(nmodels, 1) {
    curr_score.setZero(nmodels, 1);
    prev_score.setZero(nmodels, 1);
    score_trace.setZero(nmodels, 10);
  }

  template <typename Derived>
  void add(const Eigen::MatrixBase<Derived>& S) {
    update_col(S.derived());
  }

  template <typename Derived>
  void add(const Eigen::SparseMatrixBase<Derived>& S) {
    update_col(S.derived());
  }

  bool converged(const Scalar tol = 1e-4, const Index minT = 10) {
    if (t > minT && nrec >= 2) {
      const Scalar old = score_trace.col(nrec - 2).cwiseAbs().mean();
      const Scalar diff = (score_trace.col(nrec - 2) - score_trace.col(nrec - 1)).cwiseAbs().maxCoeff();
      return diff / (1e-8 + old) < tol;
    }

    return false;
  }

  void print(std::ostream& out) {
    if ((t % interv == 1) || interv < 2) {
      out << "[" << std::setw(20) << t << "]";
      out << " [" << std::setw(20) << nrec << "]";
      out << " [" << std::setw(20) << curr_score.sum() << "]";
      out << std::endl;
    }
  }

  mat_t summarize() const { return mat_t{score_trace.leftCols(nrec).transpose()}; }

 private:
  template <typename C>
  void update_col(const C& score) {
    ASSERT(score.rows() == nmodels, "different number of models");

    double_score_trace();
    curr_score = score;
    curr_score_stat(curr_score);

    if (t == 0) {
      score_trace.col(nrec) = curr_score;
      ++nrec;

    } else if (t > 0 && t % interv == 0) {
      prev_score = curr_score_stat.mean();
      score_trace.col(nrec) = prev_score;
      ++nrec;
      curr_score_stat.reset();
    }
    ++t;
  }

  void double_score_trace() {
    const int currSize = score_trace.cols();
    if (currSize <= nrec) {
      mat_t temp = score_trace;
      score_trace.resize(nmodels, 2 * currSize);
      score_trace.leftCols(nrec) = temp;
    }
  }

  int t;
  int nrec;
  const int nmodels;
  const int interv;

  running_stat_t<mat_t> curr_score_stat;
  mat_t curr_score;   // nmodel x 1
  mat_t prev_score;   // nmodel x 1
  mat_t score_trace;  // nmodel x nrec
  mat_t ones;         // temporary ndata x 1
};

#endif
