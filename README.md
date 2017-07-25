# A factorization approach to multi-tissue multi-SNP QTL modeling

NOTE: We are restructuring analysis pipeline and documents.  Results
and codes are temporary.

## Contributors

Methods are designed by Yongjin Park, Abhishek Sarkar, Kunal Bhutani, and Manolis Kellis; maintained by Yongjin Park (`ypp@csail.mit.edu`).

## Installation

Install R package:

```
library(devtools)
library(Rcpp)
install_github("ypark/fqtl")
```

C++ codes were implemented with C++14.  Please include `-std=c++14` in
`CFLAGS` and `CXXFLAGS` of `~/.R/Makevars` file.  For instance,
```
CFLAGS = -O3 -std=c++14
CXXFLAGS = -O3 -std=c++14
```

To speed up matrix-vector multiplication, one can compile codes with
Intel MKL library.  Currently `RcppEigen` does not support `BLAS` only
options.

## Models

### Gaussian model

We define mean, variance and log-likelihood:

```
E[y] = eta1,
V[y] = Vmin + (Vmax - Vmin) * sigmoid(eta2)
L[y] = -0.5 * (y - E[y])^2 / V[y] - 0.5 * ln V[y]
```

Here we stabilized error variance, restricting on [`Vmin`, `Vmax`], with `Vmax = observed V[y]` and `Vmin = 1e-4 * Vmax`.


### Negative binomial model

```
b[y] = exp(- eta1)
E[y] = exp(X theta1 + ln (1 + 1/phi))
V[y] = E[y] * (1 + E[y] / (1 + 1/phi))

L[y] = ln Gam(y + 1/phi + 1) - ln Gam(1/phi + 1)
       - y * ln(1 + exp(- eta1))
       - (1/phi + 1) * ln(1 + exp(eta1))
```

Like Gaussian model, we could bound variance model `V[y] < observed V[y]`.
Probably most loosened model fit to the data is sample mean `Y.bar` where there is no association of covariates.

```
worst_variance = Ybar * (1 + Ybar /(1 + 1/phi)) < Ybar * (1 + Ybar * phi) < Vobs
phi_max = (Vobs / Ybar - 1) / Ybar
```
