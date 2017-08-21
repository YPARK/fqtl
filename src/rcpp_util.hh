#include <Rcpp.h>
#include <cassert>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>

#ifndef RCPPUTIL_HH_
#define RCPPUTIL_HH_

std::string curr_time();

#define TLOG(msg) \
  { Rcpp::Rcerr << "[" << curr_time() << "] " << msg << std::endl; }
#define ELOG(msg) \
  { Rcpp::Rcerr << "[Error] [" << curr_time() << "] " << msg << std::endl; }
#define WLOG(msg) \
  { Rcpp::Rcerr << "[Warning] [" << curr_time() << "] " << msg << std::endl; }
#define ASSERT(cond, msg)             \
  {                                   \
    if (!(cond)) {                    \
      ELOG(msg);                      \
      Rcpp::stop("assertion failed"); \
    }                                 \
  }

#endif
