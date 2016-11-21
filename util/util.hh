#ifndef UTIL_HH_
#define UTIL_HH_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

std::string curr_time();

#define TLOG(msg) \
  { std::cerr << "         [" << curr_time() << "] " << msg << std::endl; }

#define ELOG(msg) \
  { std::cerr << "[Error  ][" << curr_time() << "] " << msg << std::endl; }

#define WLOG(msg) \
  { std::cerr << "[Warning][" << curr_time() << "] " << msg << std::endl; }

#define ASSERT(cond, msg) \
  {                       \
    if (!(cond)) {        \
      ELOG(msg);          \
      std::exit(1);       \
    }                     \
  }

#define CHK_ERR_EXIT(cond, msg)   \
  {                               \
    if ((cond) != EXIT_SUCCESS) { \
      ELOG(msg);                  \
      std::exit(1);               \
    }                             \
  }

#define CHK_ERR_RET(cond, msg)    \
  {                               \
    if ((cond) != EXIT_SUCCESS) { \
      ELOG(msg);                  \
      return EXIT_FAILURE;        \
    }                             \
  }

#define ERR_RET(cond, msg) \
  {                        \
    if ((cond)) {          \
      ELOG(msg);           \
      return EXIT_FAILURE; \
    }                      \
  }

std::string zeropad(const int t, const int tmax);

std::string curr_time() {
  time_t rawtime;
  time(&rawtime);
  struct tm* timeinfo = localtime(&rawtime);
  char buff[80];
  strftime(buff, 80, "%c", timeinfo);
  return std::string(buff);
}

std::string zeropad(const int t, const int tmax) {
  std::string tt = std::to_string(t);
  std::string ttmax = std::to_string(tmax);
  const int ndigit = ttmax.size();

  std::ostringstream ss;
  ss << std::setw(ndigit) << std::setfill('0') << tt;
  return std::string(ss.str());
}

#endif
