# A factorization approach to multi-tissue multi-SNP QTL modeling

NOTE: We are restructuring analysis pipeline and documents.  Results
and codes are temporary.

## Contributors

Methods are designed by Yongjin Park, Abhishek Sarkar, Kunal Bhutani,
and Manolis Kellis; maintained by Yongjin Park (`ypp@csail.mit.edu`).

## Installation

Install R package:

```
library(devtools)
install_github("ypark/fqtl")
```

To successfully compile the Rcpp code, `R` must have been compiled with a
compiler which supports C++14. If you get the following error when installing:

```
Error in .shlib_internal(args) : 
  C++14 standard requested but CXX14 is not defined
```

you must recompile `R` with a compiler that supports C++14 (for example, `GCC
>= 5.0`).

To speed up matrix-vector multiplication and vectorized random number
generation, one can compile the package with _Intel MKL_ or _openblas_.

## Results

Analysis pipeline and pre-calculated effect sizes computed in 44 GTEx
tissues (v6p) can be found in separate repository.

[fqtl-gtex](https://github.mit.edu/ypp/fqtl-gtex)


## Models

### Gaussian model

In Park and Sarkar _et al._ we define mean, variance and log-likelihood:

```
E[y] = eta1,
V[y] = Vmin + (Vmax - Vmin) * sigmoid(eta2)
L[y] = -0.5 * (y - E[y])^2 / V[y] - 0.5 * ln V[y]
```

Here we stabilized error variance, restricting on [`Vmin`, `Vmax`],
with `Vmax = observed V[y]` and `Vmin = 1e-4 * Vmax`.
A new version of the paper will soon be posted.

### Negative binomial model

Robinson and Smyth (2008), Anders and Huber (2010) popularized
negative binomial models for analysis of high-throughput sequencing
data.

```
b[y] = exp(- eta1)
E[y] = exp(X theta1 + ln (1 + 1/phi))
V[y] = E[y] * (1 + E[y] / (1 + 1/phi))

L[y] = ln Gam(y + 1/phi + 1) - ln Gam(1/phi + 1)
       - y * ln(1 + exp(- eta1))
       - (1/phi + 1) * ln(1 + exp(eta1))
```

Like Gaussian model, we could bound variance model `V[y] < observed
V[y]`.  Probably most loosened model fit to the data is sample mean
`Ybar`.  If there were no association of covariates, sample mean
would be unbiased estimation of rate parameter of the underlying
Poisson distribution.

```
worst_variance = Ybar * (1 + Ybar /(1 + 1/phi))
               < Ybar * (1 + Ybar * phi)
               < Vobs
```

Therefore we can claim:

```
phi_max = (Vobs / Ybar - 1) / Ybar
```


### Negative binomial model (redefined with psuedo-count)

$$p(y|\alpha,\beta) = \frac{\Gamma(\alpha + y + a_{0})}{\Gamma(y+1)\Gamma(\alpha + a_{0})}
\left(\frac{1}{1 + 1/\beta}\right)^{\alpha + a_{0}}
\left(\frac{1}{1 + \beta}\right)^{y}$$

$\mathbb{E}[y|\alpha,a_{0},\beta] = (\alpha + a_{0})/\beta$

$\mathbb{V}[y|\alpha,a_{0},\beta] = (\alpha + a_{0})/\beta + (\alpha + a_{0})/\beta^{2}$

$\mu = (a_{0} + 1/\phi) / \beta = \exp\left\{ \eta_{\mu} + \ln (a_{0} + 1/\phi) \right\}$

$\sigma^{2} = \mu + \mu^{2} / (a_{0} + \alpha)$

$\alpha = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \boldsymbol{\sigma}\!\left( - \eta_{\nu} \right)$

$\beta = \exp(-\eta_{\mu})$

---

$$\ln p(y|\alpha,\beta) = \ln \frac{\Gamma(\alpha + y + a_{0})}{\Gamma(y+1)\Gamma(\alpha + a_{0})}
- (\alpha + a_{0}) \ln \left(1 + 1/\beta\right)
- y \ln \left(1 + \beta\right)$$


$$-(\alpha + a_{0}) \ln(1 + \exp[\eta_{\mu}])$$

$$-y \ln(1 + \exp[-\eta_{\mu}])$$



---

$\bar{y} + \bar{y}^{2} / (a_{0} + \alpha) < \hat{\sigma}^{2}$

$1 / (a_{0} + \alpha) < (\hat{\sigma}^{2} / \bar{y} - 1)/ \bar{y}$

$a_{0} + \alpha > \bar{y} / (\hat{\sigma}^{2} / \bar{y} - 1)$

$\alpha > \bar{y} / (\hat{\sigma}^{2} / \bar{y} - 1) - a_{0}$

We can simply set $\alpha_{\max} = \hat{\sigma}^{2}$


### Voom model

Law _et al._ (2014) presents log2-transformed over-dispersion model.
They derived mean-variance relationship of the transformed random
variable by the Delta method.  We reparameterized the model as follows.

```
E[y] = eta1,
V[y] = exp(- eta1) + phi
phi  = phi_min + (phi_max - phi_min) * sigmoid(eta2)
```

## Statistical inference

We developed variational inference algorithm based on stochastic
gradient algorithm (Paisley _et al._ 2012).  Especially we tuned the
method to better accommodate high-dimensional generalized linear
models with data-driven sparsity.

## References

Park, Y., Sarkar, A. K., Bhutani, K., & Kellis, M. (2017). Multi-tissue polygenic models for transcriptome-wide association studies. bioRxiv, 107623. http://doi.org/10.1101/107623

Robinson, M. D., & Smyth, G. K. (2008). Small-sample estimation of negative binomial dispersion, with applications to SAGE data. Biostatistics (Oxford, England), 9(2), 321–332. http://doi.org/10.1093/biostatistics/kxm030

Anders, S., & Huber, W. (2010). Differential expression analysis for sequence count data. Genome Biology, 11(10), R106. http://doi.org/10.1186/gb-2010-11-10-r106

Law, C. W., Chen, Y., Shi, W., & Smyth, G. K. (2014). voom: Precision weights unlock linear model analysis tools for RNA-seq read counts. Genome Biology, 15(2), R29.

Paisley, J., Blei, D., & Jordan, M. (2012). Variational Bayesian Inference with Stochastic Search. In J. Langford & J. Pineau (Eds.), (pp. 1367–1374). Presented at the Proceedings of the 28th International Conference on Machine Learning, New York, NY, USA: Omnipress.
