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
compiler which supports C++14. Make sure your R development environment support `C++14` by including
`-std=c++14` to `CFLAGS` and `CXXFLAGS` in `~/.R/Makevars` file.
For instance,

```
CXX = g++-6
CXX14 = g++-6
CXXFLAGS = -O3 -std=c++14
CFLAGS = -O3 -std=c++14
```

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
E[y] = η1,
V[y] = Vmin + (Vmax - Vmin) * sigmoid(η2)
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
b[y] = exp(- η1)
E[y] = exp(Xθ + ln (1 + 1/φ))
V[y] = E[y] * (1 + E[y] / (1 + 1/φ))

L[y] = ln Gam(y + 1/φ + 1) - ln Gam(1/φ + 1)
       - y * ln(1 + exp(- η1))
       - (1/φ + 1) * ln(1 + exp(η1))
```

Like Gaussian model, we could bound variance model `V[y] < observed V[y]`.  Probably most loosened model fit to the data is sample mean `Ybar`.  If there were no association of covariates, sample mean would be unbiased estimation of rate parameter of the underlying Poisson distribution.

```
worst variance = Ybar * (1 + Ybar /(1 + 1/φ))
               < Ybar * (1 + Ybar * φ)
               < Vobs
```

Therefore we can claim that

```
φmax = (Vobs / Ybar - 1) / Ybar
```


### Negative binomial model (redefined with psuedo-count)

We define negative binomial model adding pseudo-counts.

```
p(y|α,β) = Γ(α + υ + α0)/[Γ(y+1)Γ(α + α0)] (1 + 1/β)^-(α + α0) (1 + β)^-y
```

where

```
E[y|α,α0,β] = (α + α0)/β

V[y|α,α0,β] = (α + α0)β + (α + α0)β^{2}

μ = (α0 + 1/φ) / β = exp[ η{μ} + ln (α0 + 1/φ) ]

σ^2 = μ + μ^{2} / (α0 + α)

α = αmin + (αmax - αmin) sigmoid(- η(nu))

β = exp(-η(μ))
```

### Voom model

Law _et al._ (2014) presents log2-transformed over-dispersion model.
They derived mean-variance relationship of the transformed random
variable by the Delta method.  We reparameterized the model as follows.

```
E[y] = η1,
V[y] = exp(- η1) + φ
φ  = φmin + (φmax - φmin) * sigmoid(η2)
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
