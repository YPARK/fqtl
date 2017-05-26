#ifndef OPTIONS_HH_
#define OPTIONS_HH_

#include <string>
#include <vector>

struct options_t {

  explicit options_t() {
    VBITER = 1000;
    MINITER = 100;
    K = 10;
    VBTOL = 1e-2;
    NSAMPLE = 10;
    JITTER = 1e-2;
    RATE0 = 1e-2;
    DECAY = -0.01;
    TAU_LODDS_LB = -10;
    TAU_LODDS_UB = -4;
    PI_LODDS_LB = -4;
    PI_LODDS_UB = -2;
    GAMMAX = 10000;
    INTERV = 10;
    RATE_M = 0.99;
    RATE_V = 0.9999;
    VERBOSE = true;
    // HIGH_STOCH = false;
    MODEL_NAME = "gaussian";
    OUT_RESID = true;
    RSEED = 13;
    MF_SVD_INIT = true;
  }

  const unsigned int vbiter() const { return VBITER; };
  const unsigned int miniter() const { return MINITER; };
  const unsigned int nsample() const { return NSAMPLE; };
  const unsigned int k() const { return K; };
  const float vbtol() const { return VBTOL; };
  const float jitter() const { return JITTER; };
  const float rate0() const { return RATE0; };
  const float decay() const { return DECAY; };
  const float tau_lodds_lb() const { return TAU_LODDS_LB; };
  const float tau_lodds_ub() const { return TAU_LODDS_UB; };
  const float pi_lodds_lb() const { return PI_LODDS_LB; };
  const float pi_lodds_ub() const { return PI_LODDS_UB; };
  const float gammax() const { return GAMMAX; };
  const unsigned int ninterval() const { return INTERV; };
  const float rate_m() const { return RATE_M; };
  const float rate_v() const { return RATE_V; };
  const bool verbose() const { return VERBOSE; }
  const bool out_resid() const { return OUT_RESID; }
  const bool mf_svd_init() const { return MF_SVD_INIT; }
  // const bool high_stoch() const { return HIGH_STOCH; }
  const std::string model_name() const { return MODEL_NAME; }
  const unsigned int rseed() const { return RSEED; };

  unsigned int VBITER;
  unsigned int MINITER;
  unsigned int NSAMPLE;
  unsigned int K;
  float VBTOL;
  float JITTER;
  float RATE0;
  float DECAY;

  float TAU_LODDS_LB;
  float TAU_LODDS_UB;
  float PI_LODDS_LB;
  float PI_LODDS_UB;

  float GAMMAX;
  unsigned int INTERV;
  float RATE_M;
  float RATE_V;

  bool VERBOSE;
  bool OUT_RESID;
  bool MF_SVD_INIT;
  std::string MODEL_NAME;
  unsigned int RSEED;
};

#endif
