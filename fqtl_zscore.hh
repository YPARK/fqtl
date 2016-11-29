//
// Created by Yongjin Park on 11/22/16.
//
#ifndef FQTL_FQTL_ZSCORE_HH_HH
#define FQTL_FQTL_ZSCORE_HH_HH
#include "fqtl.hh"
#include "qtl_model.hh"

struct options_t {

  explicit options_t() {
    VBITER = 1000;
    MINITER = 100;
    K = 10;
    VBTOL = 1e-4;
    NSAMPLE = 10;
    JITTER = 1e-2;
    RATE0 = 1e-2;
    DECAY = -0.01;
    TAU_LODDS_LB = -4;
    TAU_LODDS_UB = -4;
    PI_LODDS_LB = -2;
    PI_LODDS_UB = -2;
    GAMMAX = 1e4;
    INTERV = 10;
    RATE_M = 0.99;
    RATE_V = 0.9999;
    OUTPUT = "output";
  }

  std::string X_FILE;
  std::string Y_FILE;
  std::string COV_FILE;
  std::string OUTPUT;

  const int vbiter() const { return VBITER; };
  const int miniter() const { return MINITER; };
  const int nsample() const { return NSAMPLE; };
  const int k() const { return K; };
  const float vbtol() const { return VBTOL; };
  const float jitter() const { return JITTER; };
  const float rate0() const { return RATE0; };
  const float decay() const { return DECAY; };
  const float tau_lodds_lb() const { return TAU_LODDS_LB; };
  const float tau_lodds_ub() const { return TAU_LODDS_UB; };
  const float pi_lodds_lb() const { return PI_LODDS_LB; };
  const float pi_lodds_ub() const { return PI_LODDS_UB; };
  const float gammax() const { return GAMMAX; };
  const int ninterval() const { return INTERV; };
  const float rate_m() const { return RATE_M; };
  const float rate_v() const { return RATE_V; };

  int parse_command(const int argc, const char *argv[]);

 private:

  int VBITER;
  int MINITER;
  int NSAMPLE;
  int K;
  float VBTOL;
  float JITTER;
  float RATE0;
  float DECAY;

  float TAU_LODDS_LB;
  float TAU_LODDS_UB;
  float PI_LODDS_LB;
  float PI_LODDS_UB;
  float GAMMAX;
  int INTERV;
  float RATE_M;
  float RATE_V;

  template<typename T>
  T lexical_cast(const string &str) {
    T var;
    std::istringstream iss;
    iss.str(str);
    iss >> var;
    return var;
  }

};

template<typename Model, typename MeanEta, typename CovEta, typename CholKinship, typename OptT>
auto
train_regression(Model &model,
                 MeanEta &mean_eta,
                 CovEta &cov_eta,
                 const CholKinship &cholKinship,
                 const OptT &opt);

void print_help(const std::string prog);

#endif //FQTL_FQTL_ZSCORE_HH_HH

