//
// Created by Yongjin Park on 11/22/16.
//

#ifndef FQTL_QTL_MODEL_HH
#define FQTL_QTL_MODEL_HH

////////////////////////////////////////////////////////////////
template<typename T = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> >
struct normal_qtl_model_t {

  using Scalar = typename T::Scalar;
  using Index = typename T::Index;
  using Data = T;

  template<typename X>
  using Dense = Eigen::MatrixBase<X>;

  template<typename X>
  using Sparse = Eigen::SparseMatrixBase<X>;

  explicit normal_qtl_model_t(const T &yy) : n(yy.rows()), m(yy.cols()),
                                             Y(n, m),
                                             Obs(n, m),
                                             llik_mat(n, m),
                                             sampled_mat(n, m),
                                             mean_mat(n, m),
                                             var_mat(n, m),
                                             Yvar(m, 1),
                                             col_var_op(mean_mat) {

    Obs = yy.unaryExpr(obs_op);
    Y = yy.unaryExpr(rm_zero_op);
    column_var(yy, Yvar);
  }

  template<typename Derived>
  const T &eval(const Dense<Derived> &eta_mean) {
    return _eval(eta_mean.derived());
  }

  template<typename Derived>
  const T &eval(const Sparse<Derived> &eta_mean) {
    return _eval(eta_mean.derived());
  }

  template<typename Derived>
  const T &sample(const Dense<Derived> &eta_mean) {
    return _sample(eta_mean.derived());
  }

  template<typename Derived>
  const T &sample(const Sparse<Derived> &eta_mean) {
    return _sample(eta_mean.derived());
  }

  const T &llik() const { return llik_mat; }
  const Index n;
  const Index m;

  const T pve() {
    const T &obs_var_vec = col_var_op(); // ncol x 1
    return obs_var_vec.cwiseQuotient(Yvar);
  }

  const T &yvar() const {
    return Yvar;
  }

 private:
  T Y;
  T Obs;
  T llik_mat;
  T sampled_mat;
  T mean_mat;
  T var_mat;
  T Yvar;

  template<typename M>
  const T &_eval(const M &eta_mean) {
    mean_mat = eta_mean;

    const T &obs_var_vec = col_var_op(); // ncol x 1

    llik_mat = -0.5 * Y.binaryExpr(mean_mat, mean_op);

    // V[epsilon] = V[y] - V[X*theta]
    for (Index g = 0; g < m; ++g) {
      const Scalar v0 = Vmin + obs_var_vec(g);
      const Scalar vy = Yvar(g);
      Scalar vobs = ((vy - Vmin) > v0) ? (vy - v0) : Vmin;
      var_mat.col(g).setConstant(vobs);
    }

    llik_mat = llik_mat.cwiseQuotient(var_mat);
    llik_mat -= 0.5 * var_mat.unaryExpr(log_op);

    llik_mat = llik_mat.cwiseProduct(Obs);

    return llik_mat;
  }

  // sample from current eta_mean
  template<typename M>
  const T &_sample(const M &eta_mean) {
    mean_mat = eta_mean;
    const T &obs_var_vec = col_var_op(); // ncol x 1
    // V[epsilon] = V[y] - V[X*theta]
    for (Index g = 0; g < m; ++g) {
      const Scalar v0 = Vmin + obs_var_vec(g);
      const Scalar vy = Yvar(g);
      Scalar vobs = ((vy - Vmin) > v0) ? (vy - v0) : Vmin;
      var_mat.col(g).setConstant(vobs);
    }
    auto rnorm = [this](const Scalar &mu_val, const Scalar &var_val) {
      return rnorm_op(mu_val, var_val);
    };
    sampled_mat = mean_mat.binaryExpr(var_mat, rnorm);
    return sampled_mat;
  };

  is_obs_op<T> obs_op;

  // simply replace NaN with zero
  struct rm_zero_op_t {
    const Scalar operator()(const Scalar &y) const {
      return static_cast<Scalar>(std::isfinite(y) ? y : 0.0);
    }
  } rm_zero_op;

  // Squared distance between x and y
  struct dist_func_t {
    Scalar operator()(const Scalar &x, const Scalar &y) const {
      return (x - y) * (x - y);
    }
  } mean_op;

  column_var_op_t<T> col_var_op;
  log_op_t<Scalar> log_op;

  struct rnorm_op_t {
    Scalar operator()(const Scalar &mu_val, const Scalar &var_val) {
      return distrib(rng) * std::sqrt(var_val) + mu_val;
    }

    std::mt19937 rng;
    std::normal_distribution<Scalar> distrib;
  } rnorm_op;

  constexpr static Scalar Vmin = 1e-4;
};

#endif //FQTL_QTL_MODEL_HH
