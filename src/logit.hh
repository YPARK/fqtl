#ifndef LOGIT_MODEL_HH_
#define LOGIT_MODEL_HH_

template <typename T>
struct logit_model_t {
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    using Data = T;

    template <typename X>
    using Dense = Eigen::MatrixBase<X>;

    template <typename X>
    using Sparse = Eigen::SparseMatrixBase<X>;

    explicit logit_model_t(const T &yy)
        : n(yy.rows())
        , m(yy.cols())
        , Y(n, m)
        , llik_mat(n, m)
        , sampled_mat(n, m)
        , evidence_mat(n, m)
        , runif(0.0, 1.0)
    {
        alloc_memory(yy);
        evidence_mat = yy.unaryExpr(is_valid_op);
        Y = yy.unaryExpr(rm_invalid_op);
    }

    const T &llik() const { return llik_mat; }
    const Index n;
    const Index m;

    template <typename Derived, typename OtherDerived>
    const T &eval(const Dense<Derived> &eta_mean,
                  const Dense<OtherDerived> &eta_var)
    {
        return _eval(eta_mean.derived(), eta_var.derived());
    }

    template <typename Derived, typename OtherDerived>
    const T &eval(const Sparse<Derived> &eta_mean,
                  const Sparse<OtherDerived> &eta_var)
    {
        return _eval(eta_mean.derived(), eta_var.derived());
    }

    template <typename Derived, typename OtherDerived>
    const T &sample(const Dense<Derived> &eta_mean,
                    const Dense<OtherDerived> &eta_var)
    {
        return _sample(eta_mean.derived(), eta_var.derived());
    }

    template <typename Derived, typename OtherDerived>
    const T &sample(const Sparse<Derived> &eta_mean,
                    const Sparse<OtherDerived> &eta_var)
    {
        return _sample(eta_mean.derived(), eta_var.derived());
    }

private:
    T Y;
    T llik_mat;
    T sampled_mat;
    T evidence_mat;

    std::mt19937 rng;
    std::uniform_real_distribution<Scalar> runif;

    template <typename Derived>
    void alloc_memory(const Dense<Derived> &Y)
    {
        llik_mat.setZero();
        sampled_mat.setZero();
    }

    template <typename Derived>
    void alloc_memory(const Sparse<Derived> &Y)
    {
        initialize(Y, llik_mat, 0.0);
        initialize(Y, sampled_mat, 0.0);
    }

    ////////////////////////////////////////////////////////////////
    // y * eta_mean - log(1 + exp(eta_mean))
    template <typename M1, typename M2>
    const T &_eval(const M1 &eta_mean, const M2 &eta_var)
    {
        llik_mat = Y.cwiseProduct(eta_mean) - eta_mean.unaryExpr(log1pExp);
        llik_mat = llik_mat.cwiseProduct(evidence_mat);
        return llik_mat;
    }

    template <typename M1, typename M2>
    const T &_sample(const M1 &eta_mean, const M2 &eta_var)
    {
        auto rbernoulli = [this](const auto &p) {
            return runif(rng) < p ? one_val : zero_val;
        };
        sampled_mat = eta_mean.unaryExpr(sgm);
        sampled_mat = sampled_mat.unaryExpr(rbernoulli);
        return sampled_mat;
    }

    log1pExp_op_t<Scalar> log1pExp;
    sigmoid_op_t<Scalar> sgm;

    struct is_valid_op_t {
        Scalar operator()(const Scalar &y) const
        {
            if (std::isfinite(y) && y >= (-small_val) &&
                y <= (one_val + small_val))
                return one_val;
            return zero_val;
        }
    } is_valid_op;

    struct rm_invalid_op_t {
        Scalar operator()(const Scalar &y) const
        {
            if (std::isfinite(y) && y >= (-small_val) &&
                y <= (one_val + small_val))
                return y;
            return zero_val;
        }
    } rm_invalid_op;

    static constexpr Scalar one_val = 1.0;
    static constexpr Scalar zero_val = 0.0;
    static constexpr Scalar small_val = 1e-8;
};

#endif
