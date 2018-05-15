#ifndef PARAM_SPIKE_GAMMA_HH_
#define PARAM_SPIKE_GAMMA_HH_

////////////////////////////////////////////////////////////////
// To represent non-negative sparse parameters
//
// theta ~ alpha * Gamma(mu, 1) + (1 - alpha) direct
//
// mean = alpha * mu
// var = alpha * mu + alpha * (1 - alpha) mu^2
//
// alpha = sigmoid(alpha_aux)
// mu = ln(1 + exp(beta))
//
// grad_mean_alpha = mu
// grad_var_alpha = mu + (1 - 2 alpha) mu^2
//
// grad_mean_mu = alpha
// grad_var_mu = alpha + 2 alpha (1 - alpha) mu
//

template <typename T, typename S>
struct param_spike_gamma_t {
  typedef T data_t;
  typedef typename T::Scalar scalar_t;
  typedef typename T::Index index_t;
  typedef adam_t<T, scalar_t> grad_adam_t;
  typedef tag_param_spike_gamma sgd_tag;
  typedef S sparsity_tag;

  template <typename Opt>
  explicit param_spike_gamma_t(const index_t n1, const index_t n2,
                               const Opt& opt)
      : nrow(n1),
        ncol(n2),
        alpha(nrow, ncol),
        alpha_aux(nrow, ncol),
        mu(nrow, ncol),
        beta(nrow, ncol),
        theta(nrow, ncol),
        theta_var(nrow, ncol),
        grad_alpha_aux(nrow, ncol),
        grad_beta(nrow, ncol),
        pi_aux(0.0),
        r_m(opt.rate_m()),
        r_v(opt.rate_v()),
        pi_lodds_lb(opt.pi_lodds_lb()),
        pi_lodds_ub(opt.pi_lodds_ub()),
        adam_alpha_aux(r_m, r_v, nrow, ncol),
        adam_beta(r_m, r_v, nrow, ncol),
        adam_pi_aux(r_m, r_v),
        resolve_spike_op(pi_aux),
        resolve_mu_op(opt.mu_min()),
        grad_alpha_lo_op(pi_aux),
        grad_prior_pi_op(pi_val) {}

  const index_t rows() const { return nrow; }
  const index_t cols() const { return ncol; }

  const index_t nrow;
  const index_t ncol;

  T alpha;
  T alpha_aux;
  T mu;
  T beta;

  T theta;
  T theta_var;

  T grad_alpha_aux;
  T grad_beta;

  scalar_t pi_val;
  scalar_t pi_aux;
  scalar_t grad_pi_aux;

  // fixed hyperparameters
  const scalar_t r_m;
  const scalar_t r_v;
  const scalar_t pi_lodds_lb;
  const scalar_t pi_lodds_ub;

  ////////////////////////////////////////////////////////////////
  // adaptive gradient
  grad_adam_t adam_alpha_aux;
  grad_adam_t adam_beta;
  adam_t<scalar_t, scalar_t> adam_pi_aux;

  ////////////////////////////////////////////////////////////////
  // helper functors
  // sigmoid(pi_aux + alpha_aux)
  struct resolve_spike_t {
    explicit resolve_spike_t(const scalar_t& _pi_aux) : pi_aux(_pi_aux) {}
    inline const scalar_t operator()(const scalar_t& alpha_aux) const {
      const scalar_t lo = alpha_aux + pi_aux;
      if (-lo > large_exp_value) {
        return fasterexp(lo) / (one_val + fasterexp(lo));
      }
      return one_val / (one_val + fasterexp(-lo));
    }
    const scalar_t& pi_aux;
  } resolve_spike_op;

  // ln(1 + exp(beta))
  struct resolve_mu_t {
    explicit resolve_mu_t(const scalar_t _mu_min) : mu_min(_mu_min) {}
    inline const scalar_t operator()(const scalar_t& b) const {
      if (b > large_exp_value) {
        return mu_min + b;
      }
      return mu_min + fasterlog(one_val + fasterexp(b));
    }
    const scalar_t mu_min;
  } resolve_mu_op;

  ////////////////////////
  // gradient operators //
  ////////////////////////

  struct grad_alpha_lodds_t {
    explicit grad_alpha_lodds_t(const scalar_t& _lodds) : lodds(_lodds) {}
    inline const scalar_t operator()(const scalar_t& x) const {
      return lodds - x;
    }
    const scalar_t& lodds;
  } grad_alpha_lo_op;

  // (1 - 2a) mu^2 + mu
  struct grad_alpha_g2_t {
    explicit grad_alpha_g2_t() {}

    inline const scalar_t operator()(const scalar_t& a,
                                     const scalar_t& m) const {
      return (one_val - two_val * a) * m * m + m;
    }
  } grad_alpha_g2_op;

  // x * a * (1 - a)
  struct grad_alpha_chain_rule_t {
    inline const scalar_t operator()(const scalar_t& x,
                                     const scalar_t& a) const {
      return x * a * (one_val - a);
    }
  } grad_alpha_chain_op;

  // a + 2a * (1-a) mu
  struct grad_mu_g2_t {
    inline const scalar_t operator()(const scalar_t& a,
                                     const scalar_t& m) const {
      return a + two_val * a * (one_val - a) * m;
    }
  } grad_mu_g2_op;

  // x * sigmoid(b)
  struct grad_beta_chain_rule_t {
    inline const scalar_t operator()(const scalar_t& x,
                                     const scalar_t& b) const {
      if (-b > large_exp_value) {
        return x * fasterexp(b) / (one_val + fasterexp(b));
      }
      return x / (one_val + fasterexp(-b));
    }
  } grad_beta_chain_op;

  // a * mu + a * (1 - a) * mu^2
  struct theta_var_helper_op_t {
    inline const scalar_t operator()(const scalar_t& a,
                                     const scalar_t& m) const {
      return a * m + a * (one_val - a) * m * m;
    }
  } theta_var_op;

  // grad of prior wrt pi_aux
  // alpha - pi
  struct grad_prior_pi_aux_t {
    explicit grad_prior_pi_aux_t(const scalar_t& _pi_val) : pi_val(_pi_val) {}
    inline const scalar_t operator()(const scalar_t& a) const {
      return a - pi_val;
    }
    const scalar_t& pi_val;
  } grad_prior_pi_op;

  static constexpr scalar_t half_val = 0.5;
  static constexpr scalar_t one_val = 1.0;
  static constexpr scalar_t two_val = 2.0;
  static constexpr scalar_t large_exp_value = 20.0;  // exp(20) is too big
};

// clear contents
template <typename Parameter>
void impl_initialize_param(Parameter& P, const tag_param_spike_gamma) {
  setConstant(P.beta, 0.0);
  setConstant(P.mu, 0.0);
  setConstant(P.grad_alpha_aux, 0.0);
  setConstant(P.grad_beta, 0.0);

  // Start from most relaxed
  P.pi_aux = P.pi_lodds_ub;
  setConstant(P.alpha_aux, 0.0);
  P.grad_pi_aux = 0.0;
}

// factory functions
template <typename scalar_t, typename Index, typename Opt>
auto make_dense_spike_gamma(const Index n1, const Index n2, const Opt& opt) {
  using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_spike_gamma_t<Mat, tag_param_dense>;

  Param ret(n1, n2, opt);
  impl_initialize_param(ret, tag_param_spike_gamma());
  resolve_param(ret);
  resolve_hyperparam(ret);

  return ret;
}

// factory functions
template <typename scalar_t, typename Index, typename Opt>
auto make_dense_spike_gamma_ptr(const Index n1, const Index n2,
                                const Opt& opt) {
  using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_spike_gamma_t<Mat, tag_param_dense>;

  auto ret_ptr = std::make_shared<Param>(n1, n2, opt);
  Param& ret = *ret_ptr.get();
  impl_initialize_param(ret, tag_param_spike_gamma());
  resolve_param(ret);
  resolve_hyperparam(ret);

  return ret_ptr;
}

// initialize non-zeroness by adjacency A
template <typename scalar_t, typename Derived, typename Opt>
auto make_sparse_spike_gamma(const Eigen::SparseMatrixBase<Derived>& A,
                             const Opt& opt) {
  const auto n1 = A.rows();
  const auto n2 = A.cols();

  using Mat = Eigen::SparseMatrix<scalar_t, Eigen::ColMajor>;
  using Param = param_spike_gamma_t<Mat, tag_param_sparse>;

  Param ret(n1, n2, opt);
  const scalar_t eps = 1e-4;

  // just add epsilon * A to reserve spots
  initialize(A, ret.alpha, eps);
  initialize(A, ret.alpha_aux, eps);
  initialize(A, ret.mu, eps);
  initialize(A, ret.beta, eps);
  initialize(A, ret.theta, eps);
  initialize(A, ret.theta_var, eps);
  initialize(A, ret.grad_alpha_aux, eps);
  initialize(A, ret.grad_beta, eps);

  impl_initialize_param(ret, tag_param_spike_gamma());
  resolve_param(ret);
  resolve_hyperparam(ret);
  return ret;
}

// update parameters by calculated stochastic gradient
template <typename Parameter, typename scalar_t>
void impl_update_param_sgd(Parameter& P, const scalar_t rate,
                           const tag_param_spike_gamma) {
  P.alpha_aux += update_adam(P.adam_alpha_aux, P.grad_alpha_aux) * rate;
  P.beta += update_adam(P.adam_beta, P.grad_beta) * rate;
  resolve_param(P);
}

template <typename Parameter, typename scalar_t>
void impl_update_hyperparam_sgd(Parameter& P, const scalar_t rate,
                                const tag_param_spike_gamma) {
  P.pi_aux += update_adam(P.adam_pi_aux, P.grad_pi_aux) * rate;
  resolve_hyperparam(P);
}

// mean and variance
template <typename Parameter>
void impl_resolve_param(Parameter& P, const tag_param_spike_gamma) {
  P.alpha = P.alpha_aux.unaryExpr(P.resolve_spike_op);
  P.mu = P.beta.unaryExpr(P.resolve_mu_op);
  P.theta = P.alpha.cwiseProduct(P.mu);
  P.theta_var = P.alpha.binaryExpr(P.mu, P.theta_var_op);
}

template <typename Parameter>
void impl_resolve_hyperparam(Parameter& P, const tag_param_spike_gamma) {
  if (P.pi_aux > P.pi_lodds_ub) P.pi_aux = P.pi_lodds_ub;
  if (P.pi_aux < P.pi_lodds_lb) P.pi_aux = P.pi_lodds_lb;

  P.pi_val = P.resolve_spike_op(0.0);
}

template <typename Parameter, typename scalar_t, typename RNG>
void impl_perturb_param(Parameter& P, const scalar_t sd, RNG& rng,
                        const tag_param_spike_gamma) {
  std::normal_distribution<scalar_t> Norm;
  auto rnorm = [&rng, &Norm, &sd](const auto& x) { return sd * Norm(rng); };
  P.beta = P.beta.unaryExpr(rnorm);
  resolve_param(P);
}

template <typename Parameter, typename scalar_t>
void impl_perturb_param(Parameter& P, const scalar_t sd,
                        const tag_param_spike_gamma) {
  std::mt19937 rng;
  impl_perturb_param(P, sd, rng, tag_param_spike_gamma());
}

template <typename Parameter>
void impl_check_nan_param(Parameter& P, const tag_param_spike_gamma) {
  auto is_nan = [](const auto& x) { return !std::isfinite(x); };
  auto num_nan = [&is_nan](const auto& M) { return M.unaryExpr(is_nan).sum(); };
  ASSERT(num_nan(P.alpha) == 0, "found in alpha");
  ASSERT(num_nan(P.beta) == 0, "found in beta");
  ASSERT(num_nan(P.theta) == 0, "found in theta");
  ASSERT(num_nan(P.theta_var) == 0, "found in theta_var");
}

template <typename Parameter>
const auto& impl_mean_param(Parameter& P, const tag_param_spike_gamma) {
  return P.theta;
}

template <typename Parameter>
auto impl_log_odds_param(Parameter& P, const tag_param_spike_gamma) {
  return P.alpha_aux;
}

template <typename Parameter>
const auto& impl_var_param(Parameter& P, const tag_param_spike_gamma) {
  return P.theta_var;
}

////////////////////////////////////////////////////////////////
// evaluate stochastic gradient descent step
template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_param_sgd(Parameter& P, const M1& G1, const M2& G2,
                         const M3& Nobs, const tag_param_spike_gamma) {
  ////////////////////////////////////////////////////////////////
  // gradient w.r.t. alpha
  P.grad_alpha_aux =
      G1.cwiseProduct(P.mu) +  //
      G2.cwiseProduct(P.alpha.binaryExpr(P.mu, P.grad_alpha_g2_op));

  // from the hyperparameter
  P.grad_alpha_aux += P.alpha_aux.unaryExpr(P.grad_alpha_lo_op);

  // chain rule on alpha_aux
  P.grad_alpha_aux =
      P.grad_alpha_aux.binaryExpr(P.alpha, P.grad_alpha_chain_op);

  // adjust number of observations
  P.grad_alpha_aux = P.grad_alpha_aux.cwiseQuotient(Nobs);

  ////////////////////////////////////////////////////////////////
  // gradient w.r.t. mu (beta)
  P.grad_beta = G1.cwiseProduct(P.alpha) +
                G2.cwiseProduct(P.alpha.binaryExpr(P.beta, P.grad_mu_g2_op));

  // chain rule on beta
  P.grad_beta = P.grad_beta.binaryExpr(P.beta, P.grad_beta_chain_op);

  // adjust number of observations
  P.grad_beta = P.grad_beta.cwiseQuotient(Nobs);
}

template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_hyperparam_sgd(Parameter& P, const M1& G1, const M2& G2,
                              const M3& Nobs, const tag_param_spike_gamma) {
  using scalar_t = typename Parameter::scalar_t;
  const scalar_t ntot = Nobs.sum();

  ////////////////////////////////
  // gradient w.r.t. alpha
  P.grad_alpha_aux =
      G1.cwiseProduct(P.mu) +  //
      G2.cwiseProduct(P.alpha.binaryExpr(P.mu, P.grad_alpha_g2_op));

  // from the hyperparameter
  P.grad_alpha_aux += P.alpha_aux.unaryExpr(P.grad_alpha_lo_op);

  // chain rule on alpha_aux
  P.grad_alpha_aux =
      P.grad_alpha_aux.binaryExpr(P.alpha, P.grad_alpha_chain_op);

  ////////////////////////////////
  // gradient w.r.t. pi_aux
  P.grad_pi_aux = P.grad_alpha_aux.sum();
  P.grad_pi_aux += P.alpha.unaryExpr(P.grad_prior_pi_op).sum();
  P.grad_pi_aux /= ntot;

  // adjust number of observations
  P.grad_alpha_aux = P.grad_alpha_aux.cwiseQuotient(Nobs);
}

template <typename Parameter>
void impl_write_param(Parameter& P, const std::string hdr, const std::string gz,
                      const tag_param_spike_gamma) {
  write_data_file((hdr + ".theta" + gz), P.theta);
  write_data_file((hdr + ".theta_var" + gz), P.theta_var);
  typename Parameter::data_t temp =
      P.alpha_aux.unaryExpr([&P](const auto& x) { return P.pi_aux + x; });
  write_data_file((hdr + ".lodds" + gz), temp);
  write_data_file((hdr + ".spike" + gz), P.alpha);
  write_data_file((hdr + ".slab" + gz), P.mu);
}

#endif
