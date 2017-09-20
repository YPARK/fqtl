\name{Factored QTL estimation}
\alias{fqtl.regress}
\title{Factored multivariate regression used in FQTL paper}
\description{
  Estimate factored or non-factored SNP x gene / tissue polygenic
  association matrix.  More precisely, we model mean parameters of the
  Gaussian distribution by either factored mean

  \deqn{\mathsf{E}[Y] = X \theta_{\mathsf{snp}} \theta_{\mathsf{gene}}^{\top} + C \theta_{\mathsf{cov}}}{E[Y] ~ X * theta_snp * theta_gene + C * theta_cov}

  or independent mean
  \deqn{\mathsf{E}[Y] = X \theta + C \theta_{\mathsf{cov}}}{E[Y] ~ X * theta + C * theta_cov}

  and variance
  \deqn{\mathsf{V}[Y] = X_{\mathsf{var}} \theta_{\mathsf{var}}}{V[Y] ~ X.var * theta.var}

  Each element of mean coefficient matrix follows spike-slab prior;
  variance coefficients follow Gaussian distribution.
}
\usage{
fqtl.regress(y, x.mean, factored = FALSE, c.mean = NULL, x.var = NULL,
             y.loc = NULL, y.loc2 = NULL, x.mean.loc = NULL,
             c.mean.loc = NULL, cis.dist = 5e5, 
             options = list(vbiter=1000, tol=1e-8,
                           gammax=100, rate=0.01, decay=-0.1, pi.ub=-1,
                           pi.lb=-4, tau.lb=-10, tau.ub=-4,
                           verbose=TRUE))
}
\arguments{
  \item{y}{a matrix of response}
  \item{x.mean}{main design matrix for mean;
    \eqn{\eta_{\mathsf{mean}}\sim X\theta}{mean ~ x.mean * theta}}
  \item{factored}{factored mean effect or not (default: FALSE)}
  \item{c.mean}{covariate matrix for mean; \eqn{\eta_{\mathsf{mean}}
      \sim C\theta}{mean ~ c.mean * theta}}
  \item{x.var}{design matrix for variance; \eqn{\eta_{\mathsf{var}}\sim
      X\theta}{var ~ x.var * theta}}

  \item{x.mean.loc}{p x 1 (optional) genomic location of x variables}

  \item{c.mean.loc}{p x 1 (optional) genomic location of c variables}

  \item{y.loc}{p x 1 (optional) genomic location of y variables, e.g., TSS}
  \item{y.loc2}{p x 1 (optional) genomic location of y variables, e.g., TES}
  \item{cis.dist}{distance between x and y (default: 5e5)}

  \item{options}{a list of options (see below)}

  \item{do.hyper}{Hyper parameter tuning (default: FALSE)}
  \item{tau}{Fixed value of tau}
  \item{pi}{Fixed value of pi}

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
  \code{fqtl.regress()} returns list of variational parameters

  For a factored regression

  left (p x k)   : \code{mean.left$theta}, \code{mean.left$theta.var}, \code{mean.left$lodds}\cr
  right (m x k)  : \code{mean.right$theta}, \code{mean.right$theta.var}, \code{mean.right$lodds}

  For independent regressions

  mean           : \code{mean$theta}, \code{mean$theta.var}, \code{mean$lodds}

  Other covariates

  cov            : \code{mean.cov$theta}, \code{mean.cov$theta.var}, \code{mean.cov$lodds}\cr
  var            : \code{var$theta}, \code{var$theta.var}\cr
  log-likelihood : \code{llik}
}
\details{}
\seealso{fqtl.mf}
\keyword{regression, variational inference, Bayesian inference}
\references{
  Park, Sarakr, Butani, Kellis (2017)
}
\examples{
require(fqtl)
require(Matrix)

n <- 100
m <- 50
p <- 200

theta.left <- matrix(sign(rnorm(3)), 3, 1)
theta.right <- matrix(sign(rnorm(3)), 1, 3)
theta <- theta.left \%*\% theta.right

X <- matrix(rnorm(n * p), n, p)
Y <- matrix(rnorm(n * m), n, m) * 0.1
Y[,1:3] <- Y[,1:3] + X[, 1:3] \%*\% theta

## Factored regression
opt <- list(tol=1e-8, pi.ub=-1, gammax=100, vbiter=1500, out.residual=FALSE)
out <- fqtl.regress(Y, X, factored=TRUE, options = opt)
k <- dim(out$mean.left$lodds)[2]

image(Matrix(out$mean.left$theta[1:20,]))
image(Matrix(out$mean.right$theta))

## Full regression (testing sparse coeff)
out <- fqtl.regress(Y, X, factored=FALSE, y.loc=1:m, x.mean.loc=1:p, cis.dist=5, options = opt)
image(out$mean$theta[1:50,])

## Test NB regression
rho <- 1/(1 + exp(as.vector(-scale(Y))))
Y.nb <- matrix(sapply(rho, rnbinom, n = 1, size = 10), nrow = n, ncol = m)
R <- apply(log(1 + Y.nb), 2, mean)
Y.nb <- sweep(Y.nb, 2, exp(R), `/`)

opt <- list(tol=1e-8, pi.ub=-1, gammax=1e3, vbiter=1500, model = 'nb', out.residual=TRUE)
out <- fqtl.regress(Y.nb, X, factored=TRUE, options = opt)

plot(out$resid[, 1], Y.nb[, 1], pch = 19, cex = .5);

image(Matrix(out$mean.left$theta[1:20,]))
image(Matrix(out$mean.right$theta))

}
\author{Yongjin Park}
