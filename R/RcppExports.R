fqtl.mf <- function(y, x.mean = NULL, x.var = NULL, c.mean = NULL, y.loc = NULL, y.loc2 = NULL, x.mean.loc = NULL,
                    cis.dist = 5e5, options = list(vbiter=1000, tol=1e-8, gammax=100,
                                        rate=0.1, decay=-0.1, pi.ub=-1, pi.lb=-4,
                                        tau.lb=-10, tau.ub=-4, verbose=TRUE)) {
    
    n <- dim(y)[1]
    m <- dim(y)[2]

    if(is.null(x.mean)){ x.mean <- matrix(1, nrow=n, ncol=1) }
    if(is.null(x.var)){ x.var <- matrix(1, nrow=n, ncol=1) }

    stopifnot(dim(x.mean)[1] == n)
    stopifnot(dim(x.var)[1] == n)

    ## dense y ~ x.mean
    if(is.null(y.loc) || is.null(x.mean.loc)){
        return(.Call('fqtl_rcpp_train_mf', PACKAGE = 'fqtl', y, x.mean, x.var, options))
    }

    ## sparse y ~ x.mean
    p <- dim(x.mean)[2]
    stopifnot(length(x.mean.loc) == p)
    stopifnot(length(y.loc) == m)

    if(is.null(y.loc2)){
        y.loc2 <- y.loc
    } else {
        stopifnot(length(y.loc2) == m)
        y.loc2 <- pmax(y.loc, y.loc2)
    }

    cis.x.adj <- .Call('fqtl_adj', PACKAGE = 'fqtl', x.mean.loc, y.loc, y.loc2, cis.dist)

    if(!requireNamespace('Matrix', quietly = TRUE)){
        print('Matrix package is missing') 
        return(NULL)
    }

    x.adj.mean <- Matrix::sparseMatrix(i = cis.x.adj$d1, j = cis.x.adj$d2, x = 1, dims = c(p, m))

    ## without additional c.mean
    if(is.null(c.mean)){
        return(.Call('fqtl_rcpp_train_mf_cis', PACKAGE = 'fqtl', y, x.mean, x.adj.mean, x.var, options))
    }

    ## additional (dense) c.mean
    stopifnot(dim(c.mean)[1] == n)
    return(.Call('fqtl_rcpp_train_mf_cis_aux', PACKAGE = 'fqtl', y, x.mean, x.adj.mean, c.mean, x.var, options))    
}

fqtl.regress <- function(y, x.mean, factored = FALSE, c.mean = NULL, x.var = NULL,
                         y.loc = NULL, y.loc2 = NULL, x.mean.loc = NULL, c.mean.loc = NULL,
                         cis.dist = 5e5, options = list(vbiter=1000, tol=1e-8, gammax=100,
                                             rate=0.1, decay=-0.1, pi.ub=-1, pi.lb=-4,
                                             tau.lb=-10, tau.ub=-4, verbose=TRUE)) {
    
    n <- dim(y)[1]
    m <- dim(y)[2]

    if(is.null(c.mean)){ c.mean <- matrix(1, nrow=n, ncol=1) }
    if(is.null(x.var)){ x.var <- matrix(1, nrow=n, ncol=1) }

    stopifnot(dim(x.mean)[1] == n)
    stopifnot(dim(x.var)[1] == n)

    if(factored){

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

        if(!requireNamespace('Matrix', quietly = TRUE)){
            print('Matrix package is missing') 
            return(NULL)
        }

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

        if(!requireNamespace('Matrix', quietly = TRUE)){
            print('Matrix package is missing') 
            return(NULL)
        }

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

read.plink <- function(bed.header) {

    glue <- function(...) paste(..., sep='')
    fam.file <- glue(bed.header, '.fam')
    bim.file <- glue(bed.header, '.bim')
    bed.file <- glue(bed.header, '.bed')
    stopifnot(file.exists(fam.file))
    stopifnot(file.exists(bim.file))
    stopifnot(file.exists(bed.file))

    ## 1. read .fam and .bim file
    fam <- read.table(fam.file)
    bim <- read.table(bim.file)
    n <- dim(fam)[1]
    n.snp <- dim(bim)[1]

    ## 2. read .bed
    bed <- .Call('read_plink_bed', PACKAGE = 'fqtl', bed.file, n, n.snp)

    return(list(FAM=fam, BIM=bim, BED=bed))
}
