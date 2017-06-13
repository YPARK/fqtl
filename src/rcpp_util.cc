#include "rcpp_util.hh"

std::string curr_time() {
  time_t rawtime;
  time(&rawtime);
  struct tm* timeinfo = localtime(&rawtime);
  char buff[80];
  strftime(buff, 80, "%c", timeinfo);
  return std::string(buff);
}
