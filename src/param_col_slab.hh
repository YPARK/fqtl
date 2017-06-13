#ifndef PARAM_COL_SLAB_HH_
#define PARAM_COL_SLAB_HH_

template <typename T, typename S>
struct param_col_slab_t {
  typedef T data_t;
  typedef typename T::Scalar scalar_t;
  typedef typename T::Index index_t;
  typedef adam_t<T, scalar_t> grad_adam_t;
  typedef tag_param_col_slab sgd_tag;
  typedef S sparsity_tag;

  using Dense = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

  template <typename Opt>
  explicit param_col_slab_t(const index_t n1, const index_t n2, const Opt &opt)
      : nrow(n1),
        ncol(n2),
        onesN1(1, n1),
        beta(nrow, ncol),
        gamma(nrow, ncol),
        gamma_aux(nrow, ncol),
        gamma_aux_col(1, ncol),
        theta_var(nrow, ncol),
        grad_beta(nrow, ncol),
        grad_gamma_aux(nrow, ncol),
        grad_gamma_aux_col(1, ncol),
        r_m(opt.rate_m()),
        r_v(opt.rate_v()),
        tau_lodds_lb(opt.tau_lodds_lb()),
        tau_lodds_ub(opt.tau_lodds_ub()),
        adam_beta(r_m, r_v, nrow, ncol),
        adam_gamma_aux_col(r_m, r_v, 1, ncol),
        adam_tau_aux(r_m, r_v),
        resolve_prec_op(opt.gammax(), tau_aux),
        resolve_prec_prior_op(opt.gammax(), tau_aux),
        grad_beta_eb_tau_op(tau_val),
        grad_gamma_eb_tau_op(tau_val),
        grad_gamma_chain_op(tau_aux),
        grad_prior_tau_aux_op(tau_val, tau_aux) {}

  const index_t rows() const { return nrow; }
  const index_t cols() const { return ncol; }

  const index_t nrow;
  const index_t ncol;

  Dense onesN1;

  T beta;
  T gamma;
  T gamma_aux;
  T gamma_aux_col;
  T theta_var;

  T grad_beta;
  T grad_gamma_aux;
  T grad_gamma_aux_col;

  scalar_t tau_val;
  scalar_t tau_aux;
  scalar_t grad_tau_aux;

  // fixed hyperparameters
  const scalar_t r_m;
  const scalar_t r_v;
  const scalar_t tau_lodds_lb;
  const scalar_t tau_lodds_ub;

  ////////////////////////////////////////////////////////////////
  // adaptive gradient
  grad_adam_t adam_beta;
  grad_adam_t adam_gamma_aux_col;
  adam_t<scalar_t, scalar_t> adam_tau_aux;

  ////////////////////////////////////////////////////////////////
  // helper functors

  struct resolve_prec_op_t {
    explicit resolve_prec_op_t(const scalar_t _gammax, const scalar_t &_tau_aux)
        : gammax(_gammax), tau_aux(_tau_aux) {}
    const scalar_t operator()(const scalar_t &x) const {
      const scalar_t lo = x - tau_aux;
      scalar_t ret;
      if (-lo > large_exp_value) {
        ret = gammax * fasterexp(lo) / (one_val + fasterexp(lo));
      } else {
        ret = gammax / (one_val + fasterexp(-lo));
      }
      return ret + small_value;
    }
    const scalar_t gammax;
    const scalar_t &tau_aux;

    static constexpr scalar_t small_value = 1e-8;
  } resolve_prec_op;

  struct resolve_prec_prior_op_t {
    explicit resolve_prec_prior_op_t(const scalar_t _gammax,
                                     const scalar_t &_tau_aux)
        : gammax(_gammax), tau_aux(_tau_aux) {}
    const scalar_t operator()() const {
      const scalar_t lo = tau_aux;
      scalar_t ret;
      if (-lo > large_exp_value) {
        ret = gammax * fasterexp(lo) / (one_val + fasterexp(lo));
      } else {
        ret = gammax / (one_val + fasterexp(-lo));
      }
      return ret + small_value;
    }
    const scalar_t gammax;
    const scalar_t &tau_aux;
    static constexpr scalar_t small_value = 1e-8;
  } resolve_prec_prior_op;

  ////////////////////////
  // gradient operators //
  ////////////////////////

  struct grad_beta_eb_tau_t {
    explicit grad_beta_eb_tau_t(const scalar_t &_tau_val) : tau_val(_tau_val) {}
    const scalar_t operator()(const scalar_t &b) const { return tau_val * b; }
    const scalar_t &tau_val;
  } grad_beta_eb_tau_op;

  // 0.5 * (tau  / gamma - 1)
  struct grad_gamma_eb_tau_t {
    explicit grad_gamma_eb_tau_t(const scalar_t &_tau_val)
        : tau_val(_tau_val) {}
    const scalar_t operator()(const scalar_t &g) const {
      return (tau_val / g - one_val) / two_val;
    }
    const scalar_t &tau_val;
  } grad_gamma_eb_tau_op;

  // sigmoid(- gam_aux[j] - tau_aux) = 1/(1 + exp(gam_aux + tau_aux))
  struct grad_gamma_chain_rule_t {
    explicit grad_gamma_chain_rule_t(const scalar_t &_tau_aux)
        : tau_aux(_tau_aux) {}
    const scalar_t operator()(const scalar_t &g_aux) const {
      const scalar_t lo = g_aux - tau_aux;
      if (lo > large_exp_value)
        return fasterexp(-lo) / (one_val + fasterexp(-lo));
      return one_val / (one_val + fasterexp(lo));
    }
    const scalar_t &tau_aux;

  } grad_gamma_chain_op;

  // grad of prior wrt tau_aux
  // 0.5 * (1 - tau / gamma - tau * beta^2)
  struct grad_prior_tau_aux_t {
    explicit grad_prior_tau_aux_t(const scalar_t &_tau_val,
                                  const scalar_t &_tau_aux)
        : tau_val(_tau_val), tau_aux(_tau_aux) {}
    const scalar_t operator()(const scalar_t &g, const scalar_t &b) const {
      scalar_t ret = (one_val - tau_val / g - tau_val * b * b) / two_val;
      if (tau_aux > large_exp_value) {
        ret = ret * fasterexp(-tau_aux) / (one_val + fasterexp(-tau_aux));
      } else {
        ret = ret / (one_val + fasterexp(tau_aux));
      }
      return ret;
    }
    const scalar_t &tau_val;
    const scalar_t &tau_aux;

  } grad_prior_tau_aux_op;

  static constexpr scalar_t large_exp_value = 20.0;  // exp(20) is too big
  static constexpr scalar_t one_val = 1.0;
  static constexpr scalar_t two_val = 2.0;
};

// clear contents
template <typename Parameter>
void impl_initialize_param(Parameter &P, const tag_param_col_slab) {
  setConstant(P.onesN1, 1.0);
  setConstant(P.beta, 0.0);
  setConstant(P.grad_beta, 0.0);
  setConstant(P.grad_gamma_aux, 0.0);

  // Start from most relaxed
  setConstant(P.gamma_aux, P.tau_lodds_lb);
  setConstant(P.gamma_aux_col, P.tau_lodds_lb);
  P.tau_aux = P.tau_lodds_lb;
  P.grad_tau_aux = 0.0;
}

// factory functions
template <typename Scalar, typename Index, typename Opt>
auto make_dense_col_slab(const Index n1, const Index n2, const Opt &opt) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_col_slab_t<Mat, tag_param_dense>;
  Param ret(n1, n2, opt);
  impl_initialize_param(ret, tag_param_col_slab());
  resolve_param(ret);
  resolve_hyperparam(ret);
  return ret;
}

// factory functions
template <typename Scalar, typename Index, typename Opt>
auto make_dense_col_slab_ptr(const Index n1, const Index n2, const Opt &opt) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_col_slab_t<Mat, tag_param_dense>;
  auto ret_ptr = std::make_shared<Param>(n1, n2, opt);
  Param &ret = *ret_ptr.get();
  impl_initialize_param(ret, tag_param_col_slab());
  resolve_param(ret);
  resolve_hyperparam(ret);
  return ret_ptr;
}

template <typename Scalar, typename Derived, typename Opt>
auto make_sparse_col_slab(const Eigen::SparseMatrixBase<Derived> &A,
                          const Opt &opt) {
  const auto n1 = A.rows();
  const auto n2 = A.cols();

  using Mat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
  using Param = param_col_slab_t<Mat, tag_param_sparse>;
  Param ret(n1, n2, opt);

  const Scalar eps = 1e-4;

  // just add epsilon * A to reserve spots
  using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Mat Acol =
      (Dense::Ones(1, n1) * A).unaryExpr([](const auto &x) { return 1.0; });

  // just add epsilon * A to reserve spots
  initialize(A, ret.beta, eps);
  initialize(A, ret.gamma_aux, eps);
  initialize(A, ret.grad_beta, eps);
  initialize(A, ret.grad_gamma_aux, eps);

  initialize(Acol, ret.gamma_aux_col, eps);

  impl_initialize_param(ret, tag_param_col_slab());
  resolve_param(ret);
  resolve_hyperparam(ret);
  return ret;
}

// variance is not straightforward
template <typename Parameter>
void impl_resolve_param(Parameter &P, const tag_param_col_slab) {
  for (auto r = 0; r < P.rows(); ++r) {
    P.gamma_aux.row(r) = P.gamma_aux_col;
  }

  P.gamma = P.gamma_aux.unaryExpr(P.resolve_prec_op);
  P.theta_var = P.gamma.cwiseInverse();
}

template <typename Parameter>
void impl_resolve_hyperparam(Parameter &P, const tag_param_col_slab) {
  if (P.tau_aux > P.tau_lodds_ub) P.tau_aux = P.tau_lodds_ub;
  if (P.tau_aux < P.tau_lodds_lb) P.tau_aux = P.tau_lodds_lb;
  P.tau_val = P.resolve_prec_prior_op();
}

template <typename Parameter, typename Scalar, typename Rng>
void impl_perturb_param(Parameter &P, const Scalar sd, Rng &rng,
                        const tag_param_col_slab) {
  std::normal_distribution<Scalar> Norm;
  auto rnorm = [&rng, &Norm, &sd](const auto &x) { return sd * Norm(rng); };

  P.beta = P.beta.unaryExpr(rnorm);

  const auto gammax = P.resolve_prec_op.gammax;
  // gam_aux > - ln(gammax) - 2 ln(sd)
  const auto gam_aux_min = -fasterlog(gammax) - 2.0 * fasterlog(sd);
  setConstant(P.gamma_aux, gam_aux_min);

  resolve_param(P);
}

template <typename Parameter, typename Scalar>
void impl_perturb_param(Parameter &P, const Scalar sd,
                        const tag_param_col_slab) {
  std::mt19937 rng;
  impl_perturb_param(P, sd, rng, tag_param_col_slab());
}

template <typename Parameter>
const auto &impl_mean_param(Parameter &P, const tag_param_col_slab) {
  return P.beta;
}

template <typename Parameter>
const auto &impl_var_param(Parameter &P, const tag_param_col_slab) {
  return P.theta_var;
}

////////////////////////////////////////////////////////////////
// evaluate stochastic gradient descent step
template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_param_sgd(Parameter &P, const M1 &G1, const M2 &G2,
                         const M3 &Nobs, const tag_param_col_slab) {
  ////////////////////////////////
  // gradient w.r.t. beta
  P.grad_beta = G1 - P.beta.unaryExpr(P.grad_beta_eb_tau_op);

  ////////////////////////////////
  // gradient w.r.t. gamma
  // - G2 / gam - 0.5 (1 - 1/(1 + b^2 * g))
  P.grad_gamma_aux = -G2.cwiseQuotient(P.gamma);

  // prior update
  P.grad_gamma_aux -= P.gamma.unaryExpr(P.grad_gamma_eb_tau_op);

  // chain rule
  P.grad_gamma_aux = P.grad_gamma_aux.cwiseProduct(
      P.gamma_aux.unaryExpr(P.grad_gamma_chain_op));

  // adjust number of observations
  P.grad_beta = P.grad_beta.cwiseQuotient(Nobs);
  P.grad_gamma_aux = P.grad_gamma_aux.cwiseQuotient(Nobs);
}

template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_hyperparam_sgd(Parameter &P, const M1 &G1, const M2 &G2,
                              const M3 &Nobs, const tag_param_col_slab) {
  using scalar_t = typename Parameter::scalar_t;
  const scalar_t ntot = Nobs.sum();

  ////////////////////////////////
  // gradient w.r.t. gamma
  // - G2 / gam - 0.5 (1 - 1/(1 + b^2 * g))
  P.grad_gamma_aux = -G2.cwiseQuotient(P.gamma);

  // prior update
  P.grad_gamma_aux -= P.gamma.unaryExpr(P.grad_gamma_eb_tau_op);

  // chain rule
  P.grad_gamma_aux = P.grad_gamma_aux.cwiseProduct(
      P.gamma_aux.unaryExpr(P.grad_gamma_chain_op));

  ////////////////////////////////
  // gradient w.r.t. tau_aux
  P.grad_tau_aux = -P.grad_gamma_aux.sum();
  P.grad_tau_aux += P.gamma.binaryExpr(P.beta, P.grad_prior_tau_aux_op).sum();
  P.grad_tau_aux /= ntot;

  // adjust number of observations
  P.grad_gamma_aux = P.grad_gamma_aux.cwiseQuotient(Nobs);
}

// update parameters by calculated stochastic gradient
template <typename Parameter, typename Scalar>
void impl_update_param_sgd(Parameter &P, const Scalar rate,
                           const tag_param_col_slab) {
  const typename Parameter::scalar_t denom = P.rows();
  P.grad_gamma_aux_col = P.onesN1 * P.grad_gamma_aux / denom;

  P.gamma_aux_col +=
      update_adam(P.adam_gamma_aux_col, P.grad_gamma_aux_col) * rate;

  P.beta += update_adam(P.adam_beta, P.grad_beta) * rate;
  resolve_param(P);
}

template <typename Parameter, typename Scalar>
void impl_update_hyperparam_sgd(Parameter &P, const Scalar rate,
                                const tag_param_col_slab) {
  P.tau_aux += update_adam(P.adam_tau_aux, P.grad_tau_aux) * rate;
  resolve_hyperparam(P);
}

template <typename Parameter>
void impl_write_param(Parameter &P, const std::string hdr, const std::string gz,
                      const tag_param_col_slab) {
  write_data_file((hdr + ".theta" + gz), P.beta);
  write_data_file((hdr + ".theta_var" + gz), P.theta_var);

  TLOG("Tau = " << std::setw(10) << P.tau_val << " [" << std::setw(10)
                << P.tau_aux << "]");
}

template <typename Parameter>
void impl_check_nan_param(Parameter &P, const tag_param_col_slab) {
  auto is_nan = [](const auto &x) { return !std::isfinite(x); };
  auto num_nan = [&is_nan](const auto &M) { return M.unaryExpr(is_nan).sum(); };
  ASSERT(num_nan(P.beta) == 0, "found in beta");
}

#endif
