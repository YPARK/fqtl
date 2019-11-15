################################################################
#' Variational inference of (factored) regression.
#'
#' @details
#'
#' Estimate factored or non-factored SNP x gene / tissue multivariate
#' association matrix.  More precisely, we model mean parameters of
#' the Gaussian distribution by either factored mean
#'
#' \deqn{\mathsf{E}[Y] = X \theta_{\mathsf{snp}} \theta_{\mathsf{gene}}^{\top} + C \theta_{\mathsf{cov}}}{E[Y] ~ X * theta_snp * theta_gene + C * theta_cov}
#'
#' or independent mean
#' \deqn{\mathsf{E}[Y] = X \theta + C \theta_{\mathsf{cov}}}{E[Y] ~ X * theta + C * theta_cov}
#'
#' and variance
#' \deqn{\mathsf{V}[Y] = X_{\mathsf{var}} \theta_{\mathsf{var}}}{V[Y] ~ X.var * theta.var}
#'
#' Each element of mean coefficient matrix follows spike-slab prior;
#' variance coefficients follow Gaussian distribution.
#'
#' @param y [n x m] response matrix
#' @param x.mean [n x p] primary covariate matrix for mean change (can specify location)
#' @param factored is it factored regression? (default: FALSE)
#' @param k Rank of the factored model (default: 1)
#' @param svd.init Initalize by SVD (default: TRUE)
#' 
#' @param model choose an appropriate distribution for the generative model of y matrix from \code{c('gaussian', 'nb', 'logit', 'voom', 'beta')} (default: 'gaussian')
#' @param weight.nk (non-negative) weight matrix to help factors being mode interpretable
#' @param right.nn non-negativity on the right side of the factored effect (default: FALSE)
#' @param mu.min mininum non-negativity weight (default: 0.01)
#' 
#' @param c.mean [n x q] secondary covariate matrix for mean change (dense)
#' @param x.var [n x r] covariate marix for variance#'
#' @param y.loc m x 1 genomic location of y variables
#' @param y.loc2 m x 1 genomic location of y variables (secondary)
#' @param x.mean.loc p x 1 genomic location of x.mean variables
#' @param c.mean.loc q x 1 genomic location of c.mean variables
#' @param cis.dist distance cutoff between x and y
#' @param do.hyper Hyper parameter tuning (default: FALSE)
#' @param tau Fixed value of tau
#' @param pi Fixed value of pi
#' @param tau.lb Lower-bound of tau (default: -10)
#' @param tau.ub Upper-bound of tau (default: -4)
#' @param pi.lb Lower-bound of pi (default: -4)
#' @param pi.ub Upper-bound of pi (default: -1)
#' @param tol Convergence criterion (default: 1e-4)
#' @param gammax Maximum precision (default: 1000)
#' @param rate Update rate (default: 1e-2)
#' @param decay Update rate decay (default: 0)
#' @param jitter SD of random jitter for mediation & factorization (default: 0.01)
#' @param nsample Number of stochastic samples (default: 10)
#' @param vbiter Number of variational Bayes iterations (default: 2000)
#' @param verbose Verbosity (default: TRUE)
#' 
#' @param print.interv Printing interval (default: 10)
#' @param nthread Number of threads during calculation (default: 1)
#' @param rseed Random seed
#' @param options A combined list of inference/optimization options
#'
#' @return a list of variational inference results
#'
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#'
#' @examples
#'
#' require(fqtl)
#' require(Matrix)
#'
#' n <- 100
#' m <- 50
#' p <- 200
#'
#' theta.left <- matrix(sign(rnorm(3)), 3, 1)
#' theta.right <- matrix(sign(rnorm(3)), 1, 3)
#' theta <- theta.left %*% theta.right
#'
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- matrix(rnorm(n * m), n, m) * 0.1
#' Y[,1:3] <- Y[,1:3] + X[, 1:3] %*% theta
#'
#' ## Factored regression
#' opt <- list(tol=1e-8, pi.ub=-1, gammax=1e3, vbiter=1500, out.residual=FALSE, do.hyper = TRUE)
#' out <- fit.fqtl(Y, X, factored=TRUE, k = 10, options = opt)
#' k <- dim(out$mean.left$lodds)[2]
#'
#' image(Matrix(out$mean.left$theta[1:20,]))
#' image(Matrix(out$mean.right$theta))
#'
#' ## Full regression (testing sparse coeff)
#' out <- fit.fqtl(Y, X, factored=FALSE, y.loc=1:m, x.mean.loc=1:p, cis.dist=5, options = opt)
#' image(out$mean$theta[1:50,])
#'
#' ## Test NB regression
#' rho <- 1/(1 + exp(as.vector(-scale(Y))))
#' Y.nb <- matrix(sapply(rho, rnbinom, n = 1, size = 10), nrow = n, ncol = m)
#' R <- apply(log(1 + Y.nb), 2, mean)
#' Y.nb <- sweep(Y.nb, 2, exp(R), `/`)
#'
#' opt <- list(tol=1e-8, pi.ub=-1, gammax=1e3, vbiter=1500, model = 'nb', out.residual=TRUE, k = 10, do.hyper = TRUE)
#' out <- fit.fqtl(Y.nb, X, factored=TRUE, options = opt)
#'
#' image(Matrix(out$mean.left$theta[1:20,]))
#' image(Matrix(out$mean.right$theta))
#'
#' ## Simulate weighted factored regression (e.g., cell-type fraction)
#' 
#' n <- 600
#' p <- 1000
#' h2 <- 0.5
#' 
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- matrix(rnorm(n * 1), n, 1) * sqrt(1 - h2)
#' 
#' ## construct cell type specific genetic activities
#' K <- 3
#' 
#' causal <- NULL
#' eta <- matrix(nrow = n, ncol = K)
#' 
#' for(k in 1:K) {
#'   causal.k <- sample(p, 3)
#'   causal <- rbind(causal, data.frame(causal.k, k = k))
#'   eta[, k] <- eta.k <- X[, causal.k, drop = FALSE] %*% matrix(rnorm(3, 1) / sqrt(3), 3, 1)
#' }
#' 
#' ## randomly sample cell type proportions from Dirichlet
#' rdir <- function(alpha) {
#'   ret <- sapply(alpha, rbeta, n = 1, shape2 = 1)
#'   ret <- ret / sum(ret)
#'   return(ret)
#' }
#' 
#' prop <- t(sapply(1:n, function(j) rdir(alpha = rep(1, K))))
#' eta.sum <- apply(eta * prop, 1, sum)
#' Y <- Y + eta.sum * sqrt(h2)
#' 
#' opt <- list(tol=1e-8, pi = -0, gammax=1e3, vbiter=10000, out.residual = FALSE, do.hyper = FALSE)
#' out <- fit.fqtl(y = Y, x.mean = X, weight.nk = prop, right.nn = TRUE, options = opt)
#' 
#' par(mfrow = c(1, K))
#' for(.k in 1:K) {
#'   plot(out$mean.left$lodds[, .k])
#'   ck <- subset(causal, k == .k)$causal.k
#'   points(ck, out$mean.left$lodds[ck, .k], col = 2, pch = 19)
#' }
#' 
#' 
#'
#' @export
fit.fqtl <- function(y,
                     x.mean,
                     factored = FALSE,
                     svd.init = TRUE,
                     model = c('gaussian', 'nb', 'logit', 'voom', 'beta'),
                     c.mean = NULL,
                     x.var = NULL,
                     y.loc = NULL,
                     y.loc2 = NULL,
                     x.mean.loc = NULL,
                     c.mean.loc = NULL,
                     cis.dist = 1e6,
                     weight.nk = NULL,
                     do.hyper = FALSE,
                     tau = NULL,
                     pi = NULL,
                     tau.lb = -10,
                     tau.ub = -4,
                     pi.lb = -4,
                     pi.ub = -1,
                     tol = 1e-4,
                     gammax = 1e3,
                     rate = 1e-2,
                     decay = 0,
                     jitter = 1e-1,
                     nsample = 10,
                     vbiter = 2000,
                     verbose = TRUE,
                     k = 1,
                     right.nn = FALSE,
                     mu.min = 1e-2,
                     print.interv = 10,
                     nthread = 1,
                     rseed = NULL,
                     options = list()) {

    model <- match.arg(model)

    .eval <- function(txt) eval(parse(text = txt))

    ## parse options
    opt.vars <- c('do.hyper', 'tau', 'pi', 'tau.lb', 'svd.init',
                  'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                  'jitter', 'nsample', 'vbiter', 'verbose', 'k', 'mu.min', 'right.nn',
                  'print.interv', 'nthread', 'rseed', 'model')

    for(v in opt.vars) {
        val <- .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] <- val
        }
    }

    n <- nrow(y)
    m <- ncol(y)

    if(is.null(c.mean)){ c.mean <- matrix(1, nrow=n, ncol=1) }
    if(is.null(x.var)){ x.var <- matrix(1, nrow=n, ncol=1) }

    stopifnot(nrow(x.mean) == n)
    stopifnot(nrow(x.var) == n)

    if(!requireNamespace('Matrix', quietly = TRUE)){
        print('Matrix package is missing')
        return(NULL)
    }

    if(!requireNamespace('methods', quietly = TRUE)){
        print('methods package is missing')
        return(NULL)
    }

    if(!is.null(weight.nk)) {
        weighted <- TRUE
        stopifnot(nrow(weight.nk) == n)
        if(any(weight.nk < 0)) {
            print('Removed negative weights!')
            weight.nk[weight.nk < 0] <- 0
        }
    } else {
        weighted <- FALSE
    }

    if(weighted) {
        return(.Call('fqtl_rcpp_train_fwreg', PACKAGE = 'fqtl', y, x.mean, c.mean, x.var, weight.nk, options))
    } else if(factored){

        if(is.null(y.loc) || is.null(c.mean.loc)){
            return(.Call('fqtl_rcpp_train_freg', PACKAGE = 'fqtl', y, x.mean, c.mean, x.var, options))
        }

        p <- dim(c.mean)[2]
        stopifnot(length(c.mean.loc) == p)
        stopifnot(length(y.loc) == m)

        if(is.null(y.loc2)){
            y.loc2 <- y.loc
        } else {
            stopifnot(length(y.loc2) == m)
            y.loc2 <- pmax(y.loc, y.loc2)
        }

        cis.adj <- .Call('fqtl_adj', PACKAGE = 'fqtl', c.mean.loc, y.loc, y.loc2, cis.dist)

        c.mean.adj <- Matrix::sparseMatrix(i = cis.adj$d1, j = cis.adj$d2, x = 1, dims = c(p, m))

        return(.Call('fqtl_rcpp_train_freg_cis', PACKAGE = 'fqtl', y, x.mean, c.mean, c.mean.adj, x.var, options))

    } else {

        if(is.null(y.loc) || is.null(x.mean.loc)){
            return(.Call('fqtl_rcpp_train_reg', PACKAGE = 'fqtl', y, x.mean, c.mean, x.var, options))
        }

        p.x <- dim(x.mean)[2]
        stopifnot(length(x.mean.loc) == p.x)
        stopifnot(length(y.loc) == m)

        if(is.null(y.loc2)){
            y.loc2 <- y.loc
        } else {
            stopifnot(length(y.loc2) == m)
            y.loc2 <- pmax(y.loc, y.loc2)
        }

        x.cis.adj <- .Call('fqtl_adj', PACKAGE = 'fqtl', x.mean.loc, y.loc, y.loc2, cis.dist)

        x.mean.adj <- Matrix::sparseMatrix(i = x.cis.adj$d1, j = x.cis.adj$d2, x = 1, dims = c(p.x, m))

        if(is.null(c.mean.loc)){
            return(.Call('fqtl_rcpp_train_reg_cis', PACKAGE = 'fqtl', y, x.mean, x.mean.adj, c.mean, x.var, options))
        }

        ## both x.mean and c.mean hsve locations
        p.c <- dim(c.mean)[2]
        stopifnot(length(c.mean.loc) == p.c)

        c.cis.adj <- .Call('fqtl_adj', PACKAGE = 'fqtl', c.mean.loc, y.loc, y.loc2, cis.dist)
        c.mean.adj <- Matrix::sparseMatrix(i = c.cis.adj$d1, j = c.cis.adj$d2, x = 1, dims = c(p.c, m))

        return(.Call('fqtl_rcpp_train_reg_cis_cis', PACKAGE = 'fqtl',
                     y, x.mean, x.mean.adj, c.mean, c.mean.adj, x.var, options))
    }
}

################################################################
#' Variational deconvolution of matrix
#'
#' @param y [n x m] response matrix
#' @param weight.nk (non-negative) weight matrix to help factors being mode interpretable
#' @param model choose an appropriate distribution for the generative model of y matrix from \code{c('gaussian', 'nb', 'logit', 'voom', 'beta')} (default: 'gaussian')
#' @param svd.init Initalize by SVD (default: TRUE)
#' @param x.mean [n x p] covariate matrix for mean change (can specify location)
#' @param x.var [n x r] covariate marix for variance#'
#' 
#' @param do.hyper Hyper parameter tuning (default: FALSE)
#' @param tau Fixed value of tau
#' @param pi Fixed value of pi
#' @param tau.lb Lower-bound of tau (default: -10)
#' @param tau.ub Upper-bound of tau (default: -4)
#' @param pi.lb Lower-bound of pi (default: -4)
#' @param pi.ub Upper-bound of pi (default: -1)
#' @param tol Convergence criterion (default: 1e-4)
#' @param gammax Maximum precision (default: 1000)
#' @param rate Update rate (default: 1e-2)
#' @param decay Update rate decay (default: 0)
#' @param jitter SD of random jitter for mediation & factorization (default: 0.01)
#' @param nsample Number of stochastic samples (default: 10)
#' @param vbiter Number of variational Bayes iterations (default: 2000)
#' @param verbose Verbosity (default: TRUE)
#' 
#' @param right.nn non-negativity in factored effect (default: FALSE)
#' @param mu.min mininum non-negativity weight (default: 0.01)
#' @param print.interv Printing interval (default: 10)
#' @param nthread Number of threads during calculation (default: 1)
#' @param rseed Random seed
#' @param options A combined list of inference/optimization options
#'
#' @examples
#' 
#' ################################################################
#' ## Simulate weighted matrix factorization (e.g., cell-type fraction)
#' 
#' n <- 600
#' p <- 1000
#' h2 <- 0.5
#' 
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- matrix(rnorm(n * 1), n, 1) * sqrt(1 - h2)
#' 
#' ## construct cell type specific genetic activities
#' K <- 3
#' 
#' causal <- NULL
#' eta <- matrix(nrow = n, ncol = K)
#' 
#' for(k in 1:K) {
#'   causal.k <- sample(p, 3)
#'   causal <- rbind(causal, data.frame(causal.k, k = k))
#'   eta[, k] <- eta.k <- X[, causal.k, drop = FALSE] %*% matrix(rnorm(3, 1) / sqrt(3), 3, 1)
#' }
#' 
#' ## randomly sample cell type proportions from Dirichlet
#' rdir <- function(alpha) {
#'   ret <- sapply(alpha, rbeta, n = 1, shape2 = 1)
#'   ret <- ret / sum(ret)
#'   return(ret)
#' }
#' 
#' prop <- t(sapply(1:n, function(j) rdir(alpha = rep(1, K))))
#' eta.sum <- apply(eta * prop, 1, sum)
#' Y <- Y + eta.sum * sqrt(h2)
#' 
#' opt <- list(tol=1e-8, pi = -0, gammax=1e3, vbiter=10000, out.residual = FALSE,
#' do.hyper = FALSE, right.nn = TRUE)
#' out <- fit.fqtl.deconv(Y, prop, options = opt)
#'
#' y.decon <- out$U$theta
#' 
#' par(mfrow = c(1, K))
#' for(.k in 1:K) {
#'   plot(y.decon[, .k], eta[, .k], pch = 19, color = 'gray50')
#' }
#'
#' @export
#'
fit.fqtl.deconv <- function(y,
                            weight.nk,
                            svd.init = TRUE,
                            model = c('gaussian', 'nb', 'logit', 'voom', 'beta'),
                            x.mean = NULL,
                            x.var = NULL,
                            right.nn = FALSE,
                            do.hyper = FALSE,
                            tau = NULL,
                            pi = NULL,
                            tau.lb = -10,
                            tau.ub = -4,
                            pi.lb = -4,
                            pi.ub = -1,
                            tol = 1e-4,
                            gammax = 1e3,
                            rate = 1e-2,
                            decay = 0,
                            jitter = 1e-1,
                            nsample = 10,
                            vbiter = 2000,
                            verbose = TRUE,                        
                            mu.min = 1e-2,
                            print.interv = 10,
                            nthread = 1,
                            rseed = NULL,                            
                            options = list()) {


    model <- match.arg(model)

    n <- nrow(y)
    m <- ncol(y)

    if(is.null(x.mean)){ x.mean <- matrix(1, nrow=n, ncol=1) }
    if(is.null(x.var)){ x.var <- matrix(1, nrow=n, ncol=1) }

    stopifnot(nrow(weight.nk) == n)
    stopifnot(nrow(x.mean) == n)
    stopifnot(nrow(x.var) == n)

    .eval <- function(txt) eval(parse(text = txt))

    ## parse options
    opt.vars <- c('do.hyper', 'tau', 'pi', 'tau.lb', 'svd.init',
                  'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                  'jitter', 'nsample', 'vbiter', 'verbose', 'right.nn', 'mu.min',
                  'print.interv', 'nthread', 'rseed', 'model')

    for(v in opt.vars) {
        val <- .eval(v)
        if(!(v %in% names(options)) && !is.null(val)) {
            options[[v]] <- val
        }
    }

    return(.Call('fqtl_rcpp_train_deconv', y, weight.nk, x.mean, x.var,
                 options,
                 PACKAGE = 'fqtl'))
}

################################################################
#' Variational inference of matrix factorization
#'
#' @param y [n x m] response matrix
#' @param model choose an appropriate distribution for the generative model of y matrix from \code{c('gaussian', 'nb', 'logit', 'voom', 'beta')} (default: 'gaussian')
#' @param k Rank of the factorization (default: 1)
#' @param svd.init Initalize by SVD (default: TRUE)
#' @param x.mean [n x p] primary covariate matrix for mean change (can specify location)
#' @param c.mean [n x q] secondary covariate matrix for mean change (dense)
#' @param x.var [n x r] covariate marix for variance#'
#' @param y.loc m x 1 genomic location of y variables
#' @param y.loc2 m x 1 genomic location of y variables (secondary)
#' @param x.mean.loc p x 1 genomic location of x.mean variables
#' @param cis.dist distance cutoff between x and y
#' @param do.hyper Hyper parameter tuning (default: FALSE)
#' @param tau Fixed value of tau
#' @param pi Fixed value of pi
#' @param tau.lb Lower-bound of tau (default: -10)
#' @param tau.ub Upper-bound of tau (default: -4)
#' @param pi.lb Lower-bound of pi (default: -4)
#' @param pi.ub Upper-bound of pi (default: -1)
#' @param tol Convergence criterion (default: 1e-4)
#' @param gammax Maximum precision (default: 1000)
#' @param rate Update rate (default: 1e-2)
#' @param decay Update rate decay (default: 0)
#' @param jitter SD of random jitter for mediation & factorization (default: 0.01)
#' @param nsample Number of stochastic samples (default: 10)
#' @param vbiter Number of variational Bayes iterations (default: 2000)
#' @param verbose Verbosity (default: TRUE)
#' 
#' @param print.interv Printing interval (default: 10)
#' @param nthread Number of threads during calculation (default: 1)
#' @param rseed Random seed
#' @param options.mf A combined list of inference options for matrix factorization.
#' @param options.reg A combined list of inference options for regression effects.
#'
#' @return a list of variational inference results
#'
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#'
#' @details
#'
#'   Correct hidden confounders lurking in expression matrix using low-rank
#'  matrix factorization including genetic and other biological
#'  covariates.  We estimate the following model:
#'
#'  mean
#'  \deqn{\mathsf{E}[Y] = U V^{\top} + X \theta_{\mathsf{local}} + C
#'    \theta_{\mathsf{global}}}{E[Y] ~ UV' + X * theta + C * theta.c}
#'
#'  and variance
#'  \deqn{\mathsf{V}[Y] = X_{\mathsf{var}} \theta_{\mathsf{var}}}{V[Y] ~ X.var * theta.var}
#'
#'  We determined ranks by group-wise spike-slab prior
#'  imposed on the columns of U and V.
#'
#'
#' @examples
#'
#'  require(fqtl)
#'  require(Matrix)
#'  n <- 100
#'  m <- 100
#'  k <- 3
#'  p <- 200
#'  u <- matrix(rnorm(n * k), n, k)
#'  v <- matrix(rnorm(m * k), m, k)
#'  p.true <- 3
#'  theta.true <- matrix(sign(rnorm(1:p.true)), p.true, 1)
#'  X <- matrix(rnorm(n * p), n, p)
#'  y.resid <- X[,1:p.true] %*% theta.true
#'  y <- u %*% t(v) + 0.5 * matrix(rnorm(n * m), n, m)
#'  y[,1] <- y[,1] + y.resid
#'  y <- scale(y)
#'  x.v <- matrix(1, n, 1)
#'  xx <- as.matrix(cbind(X, 1))
#'  mf.opt <- list(tol=1e-8, rate=0.01, pi.ub=0, pi.lb=-2, svd.init = TRUE,
#'  jitter = 1e-1, vbiter = 1000, gammax=1e4, mf.pretrain = TRUE, k = 10)
#'  reg.opt <- list(pi.ub=-2, pi.lb=-4, gammax=1e4, vbiter = 1000)
#' 
#' ## full t(xx) * y adjacency matrix
#'  mf.out <- fit.fqtl.factorize(y, x.mean = xx, x.var = x.v, options.mf = mf.opt)
#'  image(Matrix(y), main = 'Y')
#'  image(Matrix(mf.out$U$theta), main = 'U')
#'  image(Matrix(mf.out$V$theta), main = 'V')
#'  image(Matrix(mf.out$mean$theta[1:20,]))
#' 
#' ## sparse t(xx) * y adjacency matrix
#'  mf.out <- fit.fqtl.factorize(y, x.mean = xx, x.var = x.v, x.mean.loc = 1:(p+1),
#'                    y.loc = 1:m, cis.dist = 3,
#'                    options.mf = mf.opt,
#'                    options.reg = reg.opt)
#'  image(Matrix(mf.out$U$theta), main = 'U')
#'  image(Matrix(mf.out$V$theta), main = 'V')
#'  image(Matrix(mf.out$mean$theta[1:20,]))
#' 
#' ## mixed, sparse and dense
#'  c.m <- matrix(1, n, 1)
#'  mf.out <- fit.fqtl.factorize(y, x.mean = xx, x.var = x.v, x.mean.loc = 1:(p+1),
#'                    y.loc = 1:m, cis.dist = 3, c.mean = c.m,
#'                    options.mf = mf.opt,
#'                    options.reg = reg.opt)
#'  image(Matrix(mf.out$U$theta), main = 'U')
#'  image(Matrix(mf.out$V$theta), main = 'V')
#'
#'  
#' @export
#'
fit.fqtl.factorize <- function(y,
                               k = 1,
                               svd.init = TRUE,
                               model = c('gaussian', 'nb', 'logit', 'voom', 'beta'),
                               x.mean = NULL,
                               c.mean = NULL,
                               x.var = NULL,
                               y.loc = NULL,
                               y.loc2 = NULL,
                               x.mean.loc = NULL,
                               cis.dist = 1e6,
                               do.hyper = FALSE,
                               tau = NULL,
                               pi = NULL,
                               tau.lb = -10,
                               tau.ub = -4,
                               pi.lb = -4,
                               pi.ub = -1,
                               tol = 1e-4,
                               gammax = 1e3,
                               rate = 1e-2,
                               decay = 0,
                               jitter = 1e-1,
                               nsample = 10,
                               vbiter = 2000,
                               verbose = TRUE,
                               print.interv = 10,
                               nthread = 1,
                               rseed = NULL,
                               options.mf = list(),
                               options.reg = list()){

    model <- match.arg(model)

    n <- nrow(y)
    m <- ncol(y)

    if(is.null(x.mean)){ x.mean <- matrix(1, nrow=n, ncol=1) }
    if(is.null(x.var)){ x.var <- matrix(1, nrow=n, ncol=1) }

    stopifnot(nrow(x.mean) == n)
    stopifnot(nrow(x.var) == n)

    .eval <- function(txt) eval(parse(text = txt))

    ## parse options
    opt.vars <- c('do.hyper', 'tau', 'pi', 'tau.lb', 'svd.init',
                  'tau.ub', 'pi.lb', 'pi.ub', 'tol', 'gammax', 'rate', 'decay',
                  'jitter', 'nsample', 'vbiter', 'verbose', 'k',
                  'print.interv', 'nthread', 'rseed', 'model')

    for(v in opt.vars) {
        val <- .eval(v)
        if(!(v %in% names(options.mf)) && !is.null(val)) {
            options.mf[[v]] <- val
        }
        if(!(v %in% names(options.reg)) && !is.null(val)) {
            options.reg[[v]] <- val
        }
    }

    ## Call facotrization with dense x.mean
    if(is.null(y.loc) || is.null(x.mean.loc)){
        return(.Call('fqtl_rcpp_train_mf', y, x.mean, x.var,
                     options.mf, options.reg,
                     PACKAGE = 'fqtl'))
    }

    ## sparse y ~ x.mean
    p <- ncol(x.mean)
    stopifnot(length(x.mean.loc) == p)
    stopifnot(length(y.loc) == m)

    if(is.null(y.loc2)){
        y.loc2 <- y.loc
    } else {
        stopifnot(length(y.loc2) == m)
        y.loc2 <- pmax(y.loc, y.loc2)
    }

    cis.x.adj <- .Call('fqtl_adj', x.mean.loc, y.loc, y.loc2, cis.dist,
                       PACKAGE = 'fqtl')

    if(!requireNamespace('Matrix', quietly = TRUE)){
        print('Matrix package is missing')
        return(NULL)
    }

    if(!requireNamespace('methods', quietly = TRUE)){
        print('methods package is missing')
        return(NULL)
    }

    x.adj.mean <- Matrix::sparseMatrix(i = cis.x.adj$d1,
                                       j = cis.x.adj$d2,
                                       x = 1, dims = c(p, m))

    ## without additional c.mean
    if(is.null(c.mean)){
        return(.Call('fqtl_rcpp_train_mf_cis', y, x.mean, x.adj.mean, x.var,
                     options.mf, options.reg, PACKAGE = 'fqtl'))
    }

    ## additional (dense) c.mean
    stopifnot(nrow(c.mean) == n)
    return(.Call('fqtl_rcpp_train_mf_cis_aux',
                 y, x.mean, x.adj.mean, c.mean, x.var,
                 options.mf, options.reg, PACKAGE = 'fqtl'))
}

################################################################
#' Read binary PLINK format
#' @param bed.header header for plink fileset
#' @return a list of FAM, BIM, BED data.
#' @author Yongjin Park, \email{ypp@@csail.mit.edu}, \email{yongjin.peter.park@@gmail.com}
#' @export
read.plink <- function(bed.header) {

    fam.file <- paste0(bed.header, '.fam')
    bim.file <- paste0(bed.header, '.bim')
    bed.file <- paste0(bed.header, '.bed')
    stopifnot(file.exists(fam.file))
    stopifnot(file.exists(bim.file))
    stopifnot(file.exists(bed.file))

    ## 1. read .fam and .bim file
    fam <- read.table(fam.file, header = FALSE, stringsAsFactors = FALSE)
    bim <- read.table(bim.file, header = FALSE, stringsAsFactors = FALSE)
    n <- dim(fam)[1]
    n.snp <- dim(bim)[1]

    ## 2. read .bed
    bed <- .Call('read_plink_bed', PACKAGE = 'fqtl', bed.file, n, n.snp)

    return(list(FAM=fam, BIM=bim, BED=bed))
}

################################################################
#' Normalize Negative Binomial data by row-wise geometric mean
#'
#' @param Y count data matrix
#' @return normalized count data across columns
#' @export
nb.normalize <- function(Y) {
    R <- apply(log(1 + Y), 1, mean)
    ret <- sweep(Y, 1, exp(R), `/`)
    return(ret)
}
