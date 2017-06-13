\name{Matrix factorization}
\alias{fqtl.mf}
\title{Matrix factorization used in FQTL package}
\description{
  Correct hidden confounders lurking in expression matrix using low-rank
  matrix factorization including genetic and other biological
  covariates.  We estimate the following model:

  mean
  \deqn{\mathsf{E}[Y] = U V^{\top} + X \theta_{\mathsf{local}} + C
    \theta_{\mathsf{global}}}{E[Y] ~ UV' + X * theta + C * theta.c}
  
  and variance
  \deqn{\mathsf{V}[Y] = X_{\mathsf{var}} \theta_{\mathsf{var}}}{V[Y] ~ X.var * theta.var}

  We determined ranks by group-wise spike-slab prior
  imposed on the columns of U and V.
}
\usage{
fqtl.mf(y, x.mean = NULL, x.var = NULL, c.mean = NULL,
        y.loc = NULL, y.loc2 = NULL, x.mean.loc = NULL,
        cis.dist = 5e5,
        options.mf = list())
}
\arguments{
  \item{y}{n x m response matrix}
  \item{x.mean}{n x p mean ~ x.mean * theta (primary covars)}
  \item{c.mean}{n x q mean ~ c.mean * theta.c (secondary covars)}
  \item{x.var}{n x p var ~ x.var * theta}
  \item{x.mean.loc}{p x 1 genomic location of x variables}
  \item{y.loc}{p x 1 genomic location of y variables, e.g., TSS}
  \item{y.loc2}{p x 1 genomic location of y variables, e.g., TES}
  \item{cis.dist}{distance between x and y (default: 5e5)}
  \item{options.mf}{a list of options for matrix factorization (see below)}
  \item{options.reg}{a list of options for regression (see below)}
  \item{vbiter}{ max iterations of variational inference (default: 1000)}
  \item{tol}{ tolerance level to determine convergence (default: 1e-8)}
  \item{gammax}{ maximum precision of coefficients (default: 100)}
  \item{rate}{ learning rate of SGVB (default: 0.01)}
  \item{decay}{ decay rate, i.e. rate = pow(t, decay) (default: -0.1)}
  \item{pi.lb}{ lower bound of log-odds of \eqn{\pi_{0}}{pi0} (default: -4)}
  \item{pi.ub}{ upper bound of log-odds of \eqn{\pi_{0}}{pi0} (default: -2)}
  \item{tau.lb}{ lower bound of log-odds of \eqn{\tau_{0}}{tau0} (default: -10)}
  \item{tau.ub}{ upper bound of log-odds of \eqn{\tau_{0}}{tau0} (default: -4)}
  \item{verbose}{ verbosity of SGVB updates (default: TRUE)}
}
\value{
  \code{fqtl.mf()} returns a list of variational parameters

  Random matrix factorization

  * U : \code{U$theta}, \code{U$theta.var}, \code{U$lodds}\cr
  * V : \code{V$theta}, \code{V$theta.var}, \code{V$lodds}

  Covariates association

  * mean : \code{mean$theta}, \code{mean$theta.var}, \code{mean$lodds}

  Mixed types of covariates (sparse X and dense C)

  * sparse : \code{mean.sparse$theta}, \code{mean.sparse$theta.var},
    \code{mean.sparse$lodds} \cr
  * dense : \code{mean.dense$theta}, \code{mean.dense$theta.var},
  \code{mean.dense$lodds}

  Log-likelihood :   \code{llik}

}
\details{}
\seealso{fqtl.regress}
\keyword{regression, variational inference, Bayesian inference}
\references{
  Park, Sarkar, Butani, Kellis (2017)
}
\examples{

require(fqtl)
require(Matrix)

n <- 150
m <- 100
k <- 3
p <- 200

u <- matrix(rnorm(n * k), n, k)
v <- matrix(rnorm(m * k), m, k)

p.true <- 3
theta.true <- matrix(sign(rnorm(1:p.true)), p.true, 1)
X <- matrix(rnorm(n * p), n, p)
y.resid <- X[,1:p.true] \%*\% theta.true
y <- u \%*\% t(v) + 0.5 * matrix(rnorm(n * m), n, m)
y[,1] <- y[,1] + y.resid

x.v <- matrix(1, n, 1)
xx <- as.matrix(cbind(X, 1))

mf.opt <- list(tol=1e-8, rate=0.01, pi.ub=0, pi.lb=-2, svd.init = TRUE, jitter = 1, vbiter = 2000, gammax=1000)
reg.opt <- list(pi.ub=-2, pi.lb=-4, gammax=1000)

## full t(xx) * y adjacency matrix
mf.out <- fqtl.mf(y, x.mean = xx, x.var = x.v, options.mf = mf.opt)

image(Matrix(y), main = 'Y')
image(Matrix(mf.out$U$theta), main = 'U')
image(Matrix(mf.out$V$theta), main = 'V')
image(Matrix(mf.out$mean$theta[1:20,]))

## sparse t(xx) * y adjacency matrix
mf.out <- fqtl.mf(y, x.mean = xx, x.var = x.v, x.mean.loc = 1:(p+1),
                  y.loc = 1:m, cis.dist = 3,
                  options.mf = mf.opt,
                  options.reg = reg.opt)

image(Matrix(mf.out$U$theta), main = 'U')
image(Matrix(mf.out$V$theta), main = 'V')
image(Matrix(mf.out$mean$theta[1:20,]))

## mixed, sparse and dense
c.m <- matrix(1, n, 1)
mf.out <- fqtl.mf(y, x.mean = xx, x.var = x.v, x.mean.loc = 1:(p+1),
                  y.loc = 1:m, cis.dist = 3, c.mean = c.m,
                  options.mf = mf.opt,
                  options.reg = reg.opt)

image(Matrix(mf.out$U$theta), main = 'U')
image(Matrix(mf.out$V$theta), main = 'V')
image(Matrix(mf.out$mean.sparse$theta[1:20,]))
image(Matrix(mf.out$mean.dense$theta))
}
\author{Yongjin Park}
