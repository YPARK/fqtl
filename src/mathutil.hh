#ifndef MATHUTIL_HH_
#define MATHUTIL_HH_

#include <cmath>
#include <random>
#include "fastexp.h"
#include "fastgamma.h"
#include "fastlog.h"

////////////////////////////////////////////////////////////////
template <typename T>
struct sigmoid_op_t {
  explicit sigmoid_op_t() {}
  const T operator()(const T& x) const {
    if (-x < large_value) return one_val / (one_val + fasterexp(-x));
    return fasterexp(x) / (one_val + fasterexp(x));
  }
  const T one_val = 1.0;
  const T large_value = 20.0;  // exp(20) is too big
};

template <typename T>
struct log_op_t {
  explicit log_op_t() {}
  const T operator()(const T& x) const { return fasterlog(x); }
};

template <typename T>
struct exp_op_t {
  explicit exp_op_t() {}
  const T operator()(const T& x) const { return fasterexp(x); }
};

template <typename T>
struct log1p_op_t {
  explicit log1p_op_t() {}
  const T operator()(const T& x) const { return fasterlog(1.0 + x); }
};

template <typename T>
struct inv_op_t {
  explicit inv_op_t() {}
  const T operator()(const T& x) const { return num / (x + eps); }
  const T num = 1.0;
  const T eps = 1e-8;
};

////////////////////////////////////////////////////////////////
// log(1 + exp(x)) = log(exp(x - x) + exp(x))
//                 = log(exp(x)*(1 + exp(-x)))
//                 = x + log(1 + exp(-x))
template <typename T>
struct log1pExp_op_t {
  explicit log1pExp_op_t() {}
  const T operator()(const T& x) const {
    if (x > large_value) {
      return x + fasterlog(1.0 + fasterexp(-x));
    }
    return fasterlog(1.0 + fasterexp(x));
  }
  const T large_value = 20.0;  // exp(20) is too big
};

template <typename T>
struct gammaln_op_t {
  const T operator()(const T& x) const {
#ifdef DEBUG
    ASSERT(x + TOL > 0.0f, "x must be postive in gammaln : " << x);
#endif
    return fasterlgamma(x + TOL);
  }
  constexpr static T TOL = 1e-8;
};

template <typename T>
struct clamp_abs_op_t {
  explicit clamp_abs_op_t(const T _cutoff) : cutoff(std::abs(_cutoff)) {}
  const T operator()(const T& x) const {
    T val = x;
    if (val > cutoff) val = cutoff;
    if (val < -cutoff) val = -cutoff;
    return val;
  }
  const T cutoff;
};

template <typename T>
struct clamp_op_t {
  explicit clamp_op_t(const T _lb, const T _ub) : lb(_lb), ub(_ub) {}
  const T operator()(const T& x) const {
    T val = x;
    if (val > ub) val = ub;
    if (val < lb) val = lb;
    return val;
  }
  const T lb;
  const T ub;
};

#endif
