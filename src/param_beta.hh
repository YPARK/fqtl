#ifndef PARAM_BETA_HH_
#define PARAM_BETA_HH_

////////////////////////////////////////////////////////////////
// To represent random variable between 0 and 1
//
// theta ~ Beta(mu * phi + eps, (1 - mu) * phi + eps)
//
// mean = sgm(beta)
// var = 1 / ln(1 + exp(phi_aux)) + var_min = sgm(-phi_aux) + var_min
//
// grad_mean_beta = sgm(beta) * sgm(- beta)
// grad_var_phi_aux = - sgm(phi_aux) * sgm(- phi_aux)
//                  = - (1 - var) * var

template <typename T, typename S>
struct param_beta_t {
  typedef T data_t;
  typedef typename T::Scalar scalar_t;
  typedef typename T::Index index_t;
  typedef adam_t<T, scalar_t> grad_adam_t;
  typedef tag_param_beta sgd_tag;
  typedef S sparsity_tag;

  template <typename Opt>
  explicit param_beta_t(const index_t n1, const index_t n2, const Opt& opt)
      : nrow(n1),
        ncol(n2),
        beta(nrow, ncol),
        phi(nrow, ncol),
        phi_aux(nrow, ncol),
        theta(nrow, ncol),
        theta_var(nrow, ncol),
        grad_beta(nrow, ncol),
        grad_phi_aux(nrow, ncol),
        r_m(opt.rate_m()),
        r_v(opt.rate_v()),
        var_min(opt.var_beta_min()),
        adam_beta(r_m, r_v, nrow, ncol),
        adam_phi_aux(r_m, r_v, nrow, ncol),
        resolve_var_op(var_min) {}

  const index_t rows() const { return nrow; }
  const index_t cols() const { return ncol; }

  const index_t nrow;
  const index_t ncol;

  T beta;
  T phi;
  T phi_aux;

  T theta;
  T theta_var;

  T grad_beta;
  T grad_phi_aux;

  // fixed hyperparameters
  const scalar_t r_m;
  const scalar_t r_v;
  const scalar_t var_min;

  ////////////////////////////////////////////////////////////////
  // adaptive gradient
  grad_adam_t adam_beta;
  grad_adam_t adam_phi_aux;

  ////////////////////////////////////////////////////////////////
  // helper functors
  // sigmoid(beta)
  struct resolve_mu_t {
    inline const scalar_t operator()(const scalar_t& lo) const {
      scalar_t ret;
      if (-lo > large_exp_value) {
        ret = fasterexp(lo) / (one_val + fasterexp(lo));
      }
      ret = one_val / (one_val + fasterexp(-lo));
      if (ret < tol) ret = tol;
      if (ret > (one_val - tol)) ret = one_val - tol;
      return (ret);
    }
    const scalar_t tol = 1e-8;
  } resolve_mu_op;

  // grad_mean_beta = sgm(beta) * sgm(- beta)
  struct pr_one_minus_pr_t {
    inline const scalar_t operator()(const scalar_t& pr) const {
      return pr * (one_val - pr);
    }
  } pr_one_minus_pr_op;

  // 1 / ln(1 + exp(phi_aux)) + var_min
  struct resolve_var_t {
    explicit resolve_var_t(const scalar_t _mu_min) : mu_min(_mu_min) {}
    inline const scalar_t operator()(const scalar_t& nlo) const {
      if (nlo > large_exp_value) {
        return mu_min + one_val / (one_val + fasterexp(nlo));
      }
      return mu_min + fasterexp(-nlo) / (one_val + fasterexp(-nlo));
    }
    const scalar_t mu_min;
  } resolve_var_op;

  ////////////////////////
  // gradient operators //
  ////////////////////////

  static constexpr scalar_t one_val = 1.0;
  static constexpr scalar_t large_exp_value = 20.0;  // exp(20) is too big
};

// factory functions
template <typename scalar_t, typename Index, typename Opt>
auto make_dense_beta(const Index n1, const Index n2, const Opt& opt) {
  using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_beta_t<Mat, tag_param_dense>;

  Param ret(n1, n2, opt);
  impl_initialize_param(ret, tag_param_beta());
  resolve_param(ret);
  resolve_hyperparam(ret);

  return ret;
}

// factory functions
template <typename scalar_t, typename Index, typename Opt>
auto make_dense_beta_ptr(const Index n1, const Index n2, const Opt& opt) {
  using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Param = param_beta_t<Mat, tag_param_dense>;

  auto ret_ptr = std::make_shared<Param>(n1, n2, opt);
  Param& ret = *ret_ptr.get();
  impl_initialize_param(ret, tag_param_beta());
  resolve_param(ret);
  resolve_hyperparam(ret);

  return ret_ptr;
}

// clear contents
template <typename Parameter>
void impl_initialize_param(Parameter& P, const tag_param_beta) {
  setConstant(P.beta, 0.0);
  setConstant(P.grad_beta, 0.0);
  setConstant(P.theta, 0.0);
  setConstant(P.theta_var, 0.0);
  setConstant(P.phi_aux, 0.0);
  setConstant(P.grad_phi_aux, 0.0);
}

// initialize non-zeroness by adjacency A
template <typename scalar_t, typename Derived, typename Opt>
auto make_sparse_beta(const Eigen::SparseMatrixBase<Derived>& A,
                      const Opt& opt) {
  const auto n1 = A.rows();
  const auto n2 = A.cols();

  using Mat = Eigen::SparseMatrix<scalar_t, Eigen::ColMajor>;
  using Param = param_beta_t<Mat, tag_param_sparse>;

  Param ret(n1, n2, opt);
  const scalar_t eps = 1e-4;

  // just add epsilon * A to reserve spots
  initialize(A, ret.grad_phi_aux, eps);
  initialize(A, ret.grad_beta, eps);
  initialize(A, ret.phi_aux, eps);
  initialize(A, ret.beta, eps);
  initialize(A, ret.theta, eps);
  initialize(A, ret.theta_var, eps);

  impl_initialize_param(ret, tag_param_beta());
  resolve_param(ret);
  resolve_hyperparam(ret);
  return ret;
}

// update parameters by calculated stochastic gradient
template <typename Parameter, typename scalar_t>
void impl_update_param_sgd(Parameter& P, const scalar_t rate,
                           const tag_param_beta) {
  P.beta += update_adam(P.adam_beta, P.grad_beta) * rate;
  P.phi_aux += update_adam(P.adam_phi_aux, P.grad_phi_aux) * rate;
  resolve_param(P);
}

template <typename Parameter, typename scalar_t>
void impl_update_hyperparam_sgd(Parameter& P, const scalar_t rate,
                                const tag_param_beta) {
  ;  // nothing to do
}

// mean and variance
template <typename Parameter>
void impl_resolve_param(Parameter& P, const tag_param_beta) {
  P.theta = P.beta.unaryExpr(P.resolve_mu_op);
  P.theta_var = P.phi_aux.unaryExpr(P.resolve_var_op);
}

template <typename Parameter>
void impl_resolve_hyperparam(Parameter& P, const tag_param_beta) {
  ;  // nothing to do
}

template <typename Parameter, typename scalar_t, typename RNG>
void impl_perturb_param(Parameter& P, const scalar_t sd, RNG& rng,
                        const tag_param_beta) {
  std::normal_distribution<scalar_t> Norm;
  auto rnorm = [&rng, &Norm, &sd](const auto& x) { return sd * Norm(rng); };
  P.beta = P.beta.unaryExpr(rnorm);
  resolve_param(P);
}

template <typename Parameter, typename scalar_t>
void impl_perturb_param(Parameter& P, const scalar_t sd, const tag_param_beta) {
  std::mt19937 rng;
  impl_perturb_param(P, sd, rng, tag_param_beta());
}

template <typename Parameter>
void impl_check_nan_param(Parameter& P, const tag_param_beta) {
  auto is_nan = [](const auto& x) { return !std::isfinite(x); };
  auto num_nan = [&is_nan](const auto& M) { return M.unaryExpr(is_nan).sum(); };
  ASSERT(num_nan(P.beta) == 0, "found in beta");
  ASSERT(num_nan(P.theta) == 0, "found in theta");
  ASSERT(num_nan(P.theta_var) == 0, "found in theta_var");
}

template <typename Parameter>
const auto& impl_mean_param(Parameter& P, const tag_param_beta) {
  return P.theta;
}

template <typename Parameter>
const auto& impl_var_param(Parameter& P, const tag_param_beta) {
  return P.theta_var;
}

////////////////////////////////////////////////////////////////
// evaluate stochastic gradient descent step
template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_param_sgd(Parameter& P, const M1& G1, const M2& G2,
                         const M3& Nobs, const tag_param_beta) {
  // gradient w.r.t. beta
  P.grad_beta = G1.cwiseProduct(P.theta.unaryExpr(P.pr_one_minus_pr_op))
                    .cwiseQuotient(Nobs);

  // gradient w.r.t. phi_aux
  P.grad_phi_aux = -G2.cwiseProduct(P.theta_var.unaryExpr(P.pr_one_minus_pr_op))
                        .cwiseQuotient(Nobs);
}

template <typename Parameter, typename M1, typename M2, typename M3>
void impl_eval_hyperparam_sgd(Parameter& P, const M1& G1, const M2& G2,
                              const M3& Nobs, const tag_param_beta) {
  ;  // nothing to do
}

template <typename Parameter>
void impl_write_param(Parameter& P, const std::string hdr, const std::string gz,
                      const tag_param_beta) {
  write_data_file((hdr + ".theta" + gz), P.theta);
  write_data_file((hdr + ".theta_var" + gz), P.theta_var);
}

#endif
