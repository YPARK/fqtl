#include <cmath>
#include "rcpp_util.hh"

#ifndef param_check_hh_
#define param_check_hh_

////////////////////////////////////////////////////////////////
// commonly used parameter types
template <typename T>
struct check_finite_t {
  check_finite_t(const T v) : val(v) {
    ASSERT(std::isfinite(val), "must be finite number : " << val);
  }
  const T val;
};

template <typename T>
struct check_positive_t {
  check_positive_t(const T v) : val(v) {
    ASSERT(val > zero_val, "must be positive number : " << val);
    ASSERT(std::isfinite(val), "must be finite number : " << val);
  }

  const T val;
  static constexpr T zero_val = 0.0;
};

template <typename T>
struct check_prob_t {
  check_prob_t(const T v) : val(v) {
    ASSERT(val > zero_val && val < one_val, "must be in (0, 1) : " << val);
    ASSERT(std::isfinite(val), "must be finite number : " << val);
  }

  const T val;
  static constexpr T zero_val = 0.0;
  static constexpr T one_val = 1.0;
};

#endif
