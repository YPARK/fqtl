#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cassert>
#include <Rcpp.h>

// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>

#ifndef RCPPUTIL_HH_
#define RCPPUTIL_HH_

std::string curr_time();

#define TLOG(msg) { Rcpp::Rcerr << "      [" << curr_time() << "] " << msg << std::endl; }
#define ELOG(msg) { Rcpp::Rcerr << "Error [" << curr_time() << "] " << msg << std::endl; }
#define WLOG(msg) { Rcpp::Rcerr << "Warn  [" << curr_time() << "] " << msg << std::endl; }
#define ASSERT(cond, msg) { if(!(cond)){ ELOG(msg); Rcpp::stop("assertion failed"); } }

#endif
