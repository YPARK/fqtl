#ifndef PARAM_SPIKE_SLAB_HH_
#define PARAM_SPIKE_SLAB_HH_

template <typename T, typename S>
struct param_spike_slab_t {
  typedef T data_t;
  typedef typename T::Scalar scalar_t;
  typedef typename T::Index index_t;
  typedef adam_t<T, scalar_t> grad_adam_t;
  typedef tag_param_spike_slab sgd_tag;
  typedef S sparsity_tag;

  template <typename Opt>
  explicit param_spike_slab_t(const index_t n1, const index_t n2, const Opt& opt)
      : nrow(n1),
        ncol(n2),
        alpha(nrow, ncol),
        alpha_aux(nrow, ncol),
        beta(nrow, ncol),
        gamma(nrow, ncol),
        gamma_aux(nrow, ncol),
        theta(nrow, ncol),
        theta_var(nrow, ncol),
        grad_alpha_aux(nrow, ncol),
        grad_beta(nrow, ncol),
        grad_gamma_aux(nrow, ncol),
        pi_aux(0.0),
        r_m(opt.rate_m()),
        r_v(opt.rate_v()),
        pi_lodds_lb(opt.pi_lodds_lb()),
        pi_lodds_ub(opt.pi_lodds_ub()),
        tau_lodds_lb(opt.tau_lodds_lb()),
        tau_lodds_ub(opt.tau_lodds_ub()),
        adam_alpha_aux(r_m, r_v, nrow, ncol),
        adam_beta(r_m, r_v, nrow, ncol),
        adam_gamma_aux(r_m, r_v, nrow, ncol),
        adam_pi_aux(r_m, r_v),
        adam_tau_aux(r_m, r_v),
        resolve_prec_op(opt.gammax(), tau_aux),
        resolve_prec_prior_op(opt.gammax(), tau_aux),
        resolve_spike_op(pi_aux),
        grad_alpha_lo_op(pi_aux),
        grad_alpha_eb_tau_op(tau_val),
        grad_beta_eb_tau_op(tau_val),
        grad_gamma_eb_tau_op(tau_val),
        grad_gamma_chain_op(tau_aux),
        grad_prior_pi_op(pi_val),
        grad_prior_tau_aux_op(tau_val, tau_aux) {}

  const index_t rows() const { return nrow; }
  const index_t cols() const { return ncol; }

  const index_t nrow;
  const index_t ncol;

  T alpha;
  T alpha_aux;
  T beta;
  T gamma;
  T gamma_aux;

  T theta;
  T theta_var;

  T grad_alpha_aux;
  T grad_beta;
  T grad_gamma_aux;

  scalar_t pi_val;
  scalar_t pi_aux;
  scalar_t grad_pi_aux;

  scalar_t tau_val;
  scalar_t tau_aux;
  scalar_t grad_tau_aux;

  // fixed hyperparameters
  const scalar_t r_m;
  const scalar_t r_v;
  const scalar_t pi_lodds_lb;
  const scalar_t pi_lodds_ub;
  const scalar_t tau_lodds_lb;
  const scalar_t tau_lodds_ub;

  ////////////////////////////////////////////////////////////////
  // adaptive gradient
  grad_adam_t adam_alpha_aux;
  grad_adam_t adam_beta;
  grad_adam_t adam_gamma_aux;
  adam_t<scalar_t, scalar_t> adam_pi_aux;
  adam_t<scalar_t, scalar_t> adam_tau_aux;

  ////////////////////////////////////////////////////////////////
  // helper functors
  struct resolve_prec_op_t {
    explicit resolve_prec_op_t(const scalar_t _gammax, const scalar_t& _tau_aux) : gammax(_gammax), tau_aux(_tau_aux) {}
    const scalar_t operator()(const scalar_t& gam_aux) const {
      const scalar_t lo = gam_aux - tau_aux;
      scalar_t ret;
      if (-lo > large_exp_value) {
        ret = gammax * fasterexp(lo) / (one_val + fasterexp(lo));
      } else {
        ret = gammax / (one_val + fasterexp(-lo));
      }
      return ret + small_value;
    }
    const scalar_t gammax;
    const scalar_t& tau_aux;
    static constexpr scalar_t small_value = 1e-8;
  } resolve_prec_op;

  struct resolve_prec_prior_op_t {
    explicit resolve_prec_prior_op_t(const scalar_t _gammax, const scalar_t& _tau_aux)
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
    const scalar_t& tau_aux;
    static constexpr scalar_t small_value = 1e-8;
  } resolve_prec_prior_op;

  struct resolve_spike_t {
    explicit resolve_spike_t(const scalar_t& _pi_aux) : pi_aux(_pi_aux) {}
    const scalar_t operator()(const scalar_t& alpha_aux) const {
      const scalar_t lo = alpha_aux + pi_aux;
      if (-lo > large_exp_value) {
        return fasterexp(lo) / (one_val + fasterexp(lo));
      }
      return one_val / (one_val + fasterexp(-lo));
    }
    const scalar_t& pi_aux;

  } resolve_spike_op;

  ////////////////////////
  // gradient operators //
  ////////////////////////

  struct grad_alpha_lodds_t {
    explicit grad_alpha_lodds_t(const scalar_t& _lodds) : lodds(_lodds) {}
    const scalar_t operator()(const scalar_t& x) const { return lodds - x; }
    const scalar_t& lodds;
  } grad_alpha_lo_op;

  struct grad_alpha_eb_tau_t {
    explicit grad_alpha_eb_tau_t(const scalar_t& _tau_val) : tau_val(_tau_val) {}
    const scalar_t operator()(const scalar_t& b, const scalar_t& g) const { return (b * b + one_val / g) * tau_val; }
    const scalar_t& tau_val;
  } grad_alpha_eb_tau_op;

  struct grad_alpha_chain_rule_t {
    const scalar_t operator()(const scalar_t& x, const scalar_t& a) const { return x * a * (one_val - a); }
  } grad_alpha_chain_op;

  struct grad_alpha_g2_t {
    const scalar_t operator()(const scalar_t& a, const scalar_t& b) const { return (one_val - two_val * a) * b * b; }
  } grad_alpha_g2_op;

  struct grad_beta_g2_t {
    const scalar_t operator()(const scalar_t& a, const scalar_t& b) const { return two_val * a * (one_val - a) * b; }
  } grad_beta_g2_op;

  struct grad_beta_eb_tau_t {
    explicit grad_beta_eb_tau_t(const scalar_t& _tau_val) : tau_val(_tau_val) {}
    const scalar_t operator()(const scalar_t& b) const { return tau_val * b; }
    const scalar_t& tau_val;
  } grad_beta_eb_tau_op;

  // 0.5 * (tau  / gamma - 1)
  struct grad_gamma_eb_tau_t {
    explicit grad_gamma_eb_tau_t(const scalar_t& _tau_val) : tau_val(_tau_val) {}
    const scalar_t operator()(const scalar_t& g) const { return (tau_val / g - one_val) / two_val; }
    const scalar_t& tau_val;
  } grad_gamma_eb_tau_op;

  // sigmoid(- gam_aux[j] - tau_aux) = 1/(1 + exp(gam_aux + tau_aux))
  struct grad_gamma_chain_rule_t {
    explicit grad_gamma_chain_rule_t(const scalar_t& _tau_aux) : tau_aux(_tau_aux) {}
    const scalar_t operator()(const scalar_t& g_aux) const {
      const scalar_t lo = g_aux - tau_aux;
      if (lo > large_exp_value) return fasterexp(-lo) / (one_val + fasterexp(-lo));
      return one_val / (one_val + fasterexp(lo));
    }
    const scalar_t& tau_aux;

  } grad_gamma_chain_op;

  struct theta_var_helper_op_t {
    const scalar_t operator()(const scalar_t& a, const scalar_t& b) const { return a * (one_val - a) * b * b; }
  } theta_var_op;

  // grad of prior wrt pi_aux
  // alpha - pi
  struct grad_prior_pi_aux_t {
    explicit grad_prior_pi_aux_t(const scalar_t& _pi_val) : pi_val(_pi_val) {}
    const scalar_t operator()(const scalar_t& a) const { return a - pi_val; }
    const scalar_t& pi_val;
  } grad_prior_pi_op;

  // grad of prior wrt tau_aux
  // 0.5 * (1 - tau / gamma - tau * beta^2)
  struct grad_prior_tau_aux_t {
    explicit grad_prior_tau_aux_t(const scalar_t& _tau_val, const scalar_t& _tau_aux)
        : tau_val(_tau_val), tau_aux(_tau_aux) {}
    const scalar_t operator()(const scalar_t& g, const scalar_t& b) const {
      scalar_t ret = (one_val - tau_val / g - tau_val * b * b) / two_val;
      if (tau_aux > large_exp_value) {
        ret = ret * fasterexp(-tau_aux) / (one_val + fasterexp(-tau_aux));
      } else {
        ret = ret / (one_val + fasterexp(tau_aux));
      }
      return ret;
    }
    const scalar_t& tau_val;
    const scalar_t& tau_aux;
  } grad_prior_tau_aux_op;

  static constexpr scalar_t one_val = 1.0;
  static constexpr scalar_t two_val = 2.0;
  static constexpr scalar_t large_exp_value = 20.0;  // exp(20) is too big
};

// clear contents
template <typename Parameter>
void impl_initialize_param(Parameter& P, const tag_param_spike_slab) {
  setConstant(P.beta, 0.0);
  setConstant(P.grad_alpha_aux, 0.0);
  setConstant(P.grad_beta, 0.0);
  setConstant(P.grad_gamma_aux, 0.0);

  // Start from most relaxed
  P.pi_aux = P.pi_lodds_ub;
  setConstant(P.alpha_aux, 0.0);
  P.grad_pi_aux = 0.0;

  // complementary to each other
  setConstant(P.gamma_aux, P.tau_lodds_lb);
  P.tau_aux = P.tau_lodds_lb;
  P.grad_tau_aux = 0.0;
}

// factory functions
template <typename scalar_t, typename Index, typename Opt>
auto make_dense_spike_slab(const Index n1, const Index n2, const Opt& opt) {
  using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_spike_slab_t<Mat, tag_param_dense>;

  Param ret(n1, n2, opt);
  impl_initialize_param(ret, tag_param_spike_slab());
  resolve_param(ret);
  resolve_hyperparam(ret);

  return ret;
}

// factory functions
template <typename scalar_t, typename Index, typename Opt>
auto make_dense_spike_slab_ptr(const Index n1, const Index n2, const Opt& opt) {
  using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_spike_slab_t<Mat, tag_param_dense>;

  auto ret_ptr = std::make_shared<Param>(n1, n2, opt);
  Param& ret = *ret_ptr.get();
  impl_initialize_param(ret, tag_param_spike_slab());
  resolve_param(ret);
  resolve_hyperparam(ret);

  return ret_ptr;
}

// initialize non-zeroness by adjacency A
template <typename scalar_t, typename Derived, typename Opt>
auto make_sparse_spike_slab(const Eigen::SparseMatrixBase<Derived>& A, const Opt& opt) {
  const auto n1 = A.rows();
  const auto n2 = A.cols();

  using Mat = Eigen::SparseMatrix<scalar_t, Eigen::ColMajor>;
  using Param = param_spike_slab_t<Mat, tag_param_sparse>;

  Param ret(n1, n2, opt);
  const scalar_t eps = 1e-4;

  // just add epsilon * A to reserve spots
  initialize(A, ret.alpha, eps);
  initialize(A, ret.alpha_aux, eps);
  initialize(A, ret.beta, eps);
  initialize(A, ret.gamma, eps);
  initialize(A, ret.theta, eps);
  initialize(A, ret.theta_var, eps);
  initialize(A, ret.gamma_aux, eps);
  initialize(A, ret.grad_alpha_aux, eps);
  initialize(A, ret.grad_beta, eps);
  initialize(A, ret.grad_gamma_aux, eps);

  impl_initialize_param(ret, tag_param_spike_slab());
  resolve_param(ret);
  resolve_hyperparam(ret);
  return ret;
}

// update parameters by calculated stochastic gradient
template <typename Parameter, typename scalar_t>
void impl_update_param_sgd(Parameter& P, const scalar_t rate, const tag_param_spike_slab) {
  P.alpha_aux += update_adam(P.adam_alpha_aux, P.grad_alpha_aux) * rate;
  P.beta += update_adam(P.adam_beta, P.grad_beta) * rate;
  P.gamma_aux += update_adam(P.adam_gamma_aux, P.grad_gamma_aux) * rate;
  resolve_param(P);
}

template <typename Parameter, typename scalar_t>
void impl_update_hyperparam_sgd(Parameter& P, const scalar_t rate, const tag_param_spike_slab) {
  P.pi_aux += update_adam(P.adam_pi_aux, P.grad_pi_aux) * rate;
  P.tau_aux += update_adam(P.adam_tau_aux, P.grad_tau_aux) * rate;
  resolve_hyperparam(P);
}

// mean and variance
template <typename Parameter>
void impl_resolve_param(Parameter& P, const tag_param_spike_slab) {
  P.alpha = P.alpha_aux.unaryExpr(P.resolve_spike_op);
  P.theta = P.alpha.cwiseProduct(P.beta);
  P.gamma = P.gamma_aux.unaryExpr(P.resolve_prec_op);
  P.theta_var = P.alpha.cwiseQuotient(P.gamma) + P.alpha.binaryExpr(P.beta, P.theta_var_op);
}

template <typename Parameter>
void impl_resolve_hyperparam(Parameter& P, const tag_param_spike_slab) {
  if (P.pi_aux > P.pi_lodds_ub) P.pi_aux = P.pi_lodds_ub;
  if (P.pi_aux < P.pi_lodds_lb) P.pi_aux = P.pi_lodds_lb;

  if (P.tau_aux > P.tau_lodds_ub) P.tau_aux = P.tau_lodds_ub;
  if (P.tau_aux < P.tau_lodds_lb) P.tau_aux = P.tau_lodds_lb;

  P.pi_val = P.resolve_spike_op(0.0);
  P.tau_val = P.resolve_prec_prior_op();
}

template <typename Parameter, typename scalar_t>
void impl_perturb_param(Parameter& P, const scalar_t sd, const tag_param_spike_slab) {
  std::mt19937 rng;
  std::normal_distribution<scalar_t> Norm;
  auto rnorm = [&rng, &Norm, &sd](const auto& x) { return sd * Norm(rng); };

  P.beta = P.beta.unaryExpr(rnorm);

  const auto gammax = P.resolve_prec_op.gammax;
  // gam_aux > - ln(gammax) - 2 ln(sd)
  const auto gam_aux_min = -fasterlog(gammax) - 2.0 * fasterlog(sd);
  setConstant(P.gamma_aux, gam_aux_min);

  resolve_param(P);
}

template <typename Parameter>
void impl_check_nan_param(Parameter& P, const tag_param_spike_slab) {
  auto is_nan = [](const auto& x) { return !std::isfinite(x); };
  auto num_nan = [&is_nan](const auto& M) { return M.unaryExpr(is_nan).sum(); };
  ASSERT(num_nan(P.alpha) == 0, "found in alpha");
  ASSERT(num_nan(P.beta) == 0, "found in beta");
  ASSERT(num_nan(P.theta) == 0, "found in theta");
  ASSERT(num_nan(P.theta_var) == 0, "found in theta_var");
}

template <typename Parameter>
const auto& impl_mean_param(Parameter& P, const tag_param_spike_slab) {
  return P.theta;
}

template <typename Parameter>
auto impl_log_odds_param(Parameter& P, const tag_param_spike_slab) {
  return P.alpha_aux.unaryExpr([&P](const auto& x) { return P.pi_aux + x; });
}

template <typename Parameter>
const auto& impl_var_param(Parameter& P, const tag_param_spike_slab) {
  return P.theta_var;
}

////////////////////////////////////////////////////////////////
// evaluate stochastic gradient descent step
template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_param_sgd(Parameter& P, const M1& G1, const M2& G2, const M3& Nobs, const tag_param_spike_slab) {
  ////////////////////////////////
  // gradient w.r.t. alpha
  P.grad_alpha_aux = G1.cwiseProduct(P.beta) +
                     G2.cwiseProduct(P.gamma.cwiseInverse() + P.alpha.binaryExpr(P.beta, P.grad_alpha_g2_op));

  // prior udpate with adpative tau
  P.grad_alpha_aux -= 0.5 * P.beta.binaryExpr(P.gamma, P.grad_alpha_eb_tau_op);

  P.grad_alpha_aux += P.alpha_aux.unaryExpr(P.grad_alpha_lo_op);

  // chain rule on alpha_aux
  P.grad_alpha_aux = P.grad_alpha_aux.binaryExpr(P.alpha, P.grad_alpha_chain_op);

  // adjust number of observations
  P.grad_alpha_aux = P.grad_alpha_aux.cwiseQuotient(Nobs);

  ////////////////////////////////
  // gradient w.r.t. beta
  P.grad_beta = G1.cwiseProduct(P.alpha) + G2.cwiseProduct(P.alpha.binaryExpr(P.beta, P.grad_beta_g2_op));

  // prior update
  P.grad_beta -= P.alpha.cwiseProduct(P.beta.unaryExpr(P.grad_beta_eb_tau_op));

  // adjust number of observations
  P.grad_beta = P.grad_beta.cwiseQuotient(Nobs);

  ////////////////////////////////
  // gradient w.r.t. gamma_aux
  // - G2 / gam * sgm(-gau_aux) * alpha
  P.grad_gamma_aux = -G2.cwiseQuotient(P.gamma);

  // prior update
  P.grad_gamma_aux -= P.gamma.unaryExpr(P.grad_gamma_eb_tau_op);

  // chain rule
  P.grad_gamma_aux = P.grad_gamma_aux.cwiseProduct(P.alpha).cwiseProduct(P.gamma_aux.unaryExpr(P.grad_gamma_chain_op));

  // adjust number of observations
  P.grad_gamma_aux = P.grad_gamma_aux.cwiseQuotient(Nobs);
}

template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_hyperparam_sgd(Parameter& P, const M1& G1, const M2& G2, const M3& Nobs, const tag_param_spike_slab) {
  using scalar_t = typename Parameter::scalar_t;
  const scalar_t ntot = Nobs.sum();

  ////////////////////////////////
  // gradient w.r.t. alpha
  P.grad_alpha_aux = G1.cwiseProduct(P.beta) +
                     G2.cwiseProduct(P.gamma.cwiseInverse() + P.alpha.binaryExpr(P.beta, P.grad_alpha_g2_op));

  // prior udpate with adpative tau
  P.grad_alpha_aux -= 0.5 * P.beta.binaryExpr(P.gamma, P.grad_alpha_eb_tau_op);

  P.grad_alpha_aux += P.alpha_aux.unaryExpr(P.grad_alpha_lo_op);

  // chain rule on alpha_aux
  P.grad_alpha_aux = P.grad_alpha_aux.binaryExpr(P.alpha, P.grad_alpha_chain_op);

  ////////////////////////////////
  // gradient w.r.t. pi_aux
  P.grad_pi_aux = P.grad_alpha_aux.sum();
  P.grad_pi_aux += P.alpha.unaryExpr(P.grad_prior_pi_op).sum();
  P.grad_pi_aux /= ntot;

  // adjust number of observations
  P.grad_alpha_aux = P.grad_alpha_aux.cwiseQuotient(Nobs);

  ////////////////////////////////
  // gradient w.r.t. gamma_aux
  // - G2 / gam * sgm(-gau_aux) * alpha
  P.grad_gamma_aux = -G2.cwiseQuotient(P.gamma);

  // prior update
  P.grad_gamma_aux -= P.gamma.unaryExpr(P.grad_gamma_eb_tau_op);

  // chain rule
  P.grad_gamma_aux = P.grad_gamma_aux.cwiseProduct(P.alpha).cwiseProduct(P.gamma_aux.unaryExpr(P.grad_gamma_chain_op));

  ////////////////////////////////
  // gradient w.r.t. tau_aux
  P.grad_tau_aux = -P.grad_gamma_aux.sum();
  P.grad_tau_aux += P.alpha.cwiseProduct(P.gamma.binaryExpr(P.beta, P.grad_prior_tau_aux_op)).sum();
  P.grad_tau_aux /= ntot;

  // adjust number of observations
  P.grad_gamma_aux = P.grad_gamma_aux.cwiseQuotient(Nobs);
}

template <typename Parameter>
void impl_write_param(Parameter& P, const std::string hdr, const std::string gz, const tag_param_spike_slab) {
  write_data_file((hdr + ".theta" + gz), P.theta);
  write_data_file((hdr + ".theta_var" + gz), P.theta_var);
  typename Parameter::data_t temp = P.alpha_aux.unaryExpr([&P](const auto& x) { return P.pi_aux + x; });
  write_data_file((hdr + ".lodds" + gz), temp);
  write_data_file((hdr + ".spike" + gz), P.alpha);
  write_data_file((hdr + ".slab" + gz), P.beta);
  write_data_file((hdr + ".slab_prec" + gz), P.gamma);

  TLOG("Pi = " << std::setw(10) << P.pi_val << " [" << std::setw(10) << P.pi_aux << "] Tau = " << std::setw(10)
               << P.tau_val << " [" << std::setw(10) << P.tau_aux << "]");
}

#endif
