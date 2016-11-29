#include "fqtl_zscore.hh"

template<typename Model, typename MeanEta, typename CovEta, typename CholKinship, typename OptT>
auto
train_regression(Model &model,
                 MeanEta &mean_eta,
                 CovEta &cov_eta,
                 const CholKinship &cholKinship,
                 const OptT &opt) {

  using Scalar = typename MeanEta::Scalar;
  using Index = typename MeanEta::Index;
  using Mat = typename MeanEta::ReprMatrix;

  const Index n = model.n;
  const Index m = model.m;
  const Index ninterv = opt.ninterval();

  using Prog = progress_t<Scalar>;
  Prog progress((typename Prog::Nmodels(m)), (typename Prog::Interv(ninterv)));
  Mat onesN(model.llik().rows(), 1);
  onesN.setOnes();

  const int S = opt.nsample();

  std::mt19937 rng;
  std::normal_distribution<Scalar> Norm;
  auto rand_norm = [&rng, &Norm] { return Norm(rng); };

  // No tissue-tissue correlation yet
  dummy_rotation_t cholTis;
  dummy_rotation_t cholKinshipDummy;

  TLOG("Start training regression");
  int t;
  for (t = 0; t < opt.vbiter(); ++t) {
    const Scalar rate = opt.rate0() * std::pow(static_cast<Scalar>(t + 1), opt.decay());
    update_eta(model, mean_eta, cov_eta.repr_mean(), cholKinship, cholTis, rand_norm, S, rate);
    update_eta(model, cov_eta, mean_eta.repr_mean(), cholKinshipDummy, cholTis, rand_norm, S, rate);

    model.eval(mean_eta.repr_mean() + cov_eta.repr_mean());

    progress.add(model.pve());

    bool converged = progress.converged(opt.vbtol(), opt.miniter());
    if (converged) {
      TLOG("Converged");
      break;
    }
    progress.print(std::cerr);
  }
  TLOG("Finished training regression");

  auto &yvar = model.yvar();
  Mat etavar(model.m, 1);
  Mat pve(model.m, 1);
  column_var(mean_eta.repr_mean(), etavar);
  pve = etavar.cwiseQuotient(yvar);
  return pve;
}

void print_help(const std::string prog) {
  std::cerr << prog << " -y phenotypes -x genotypes -cov covariates -o output"
            << std::endl;
  std::cerr << std::endl;
  std::cerr << " computes z-score matrix using X and Y matrices." << std::endl;
  std::cerr << std::endl;
  std::cerr << " -pi_lodds and -tau_lodds set hyperparameters" << std::endl;
  std::cerr << " -gammax sets hyperparameter for maximum precision" << std::endl;
  std::cerr << " PIP       = sigmoid(prior_pi_lodds + local_lodds)" << std::endl;
  std::cerr << " Precision = gammax * sigmoid(local_lodds - prior_tau_lodds)" << std::endl;
  std::cerr << std::endl;
}

int options_t::parse_command(const int argc, const char **argv) {

  if (argc < 5)
    return EXIT_FAILURE;

  for (int pos = 1; pos < argc; ++pos) {
    std::string curr = argv[pos];

    if (curr == "-x" && (++pos) < argc) {
      X_FILE = argv[pos];
    } else if (curr == "-y" && (++pos) < argc) {
      Y_FILE = argv[pos];
    } else if (curr == "-cov" && (++pos) < argc) {
      COV_FILE = argv[pos];
    } else if (curr == "-o" && (++pos) < argc) {
      OUTPUT = argv[pos];
    } else if (curr == "-vbiter" && (++pos) < argc) {
      VBITER = lexical_cast<int>(argv[pos]);
    } else if (curr == "-k" && (++pos) < argc) {
      K = lexical_cast<int>(argv[pos]);
    } else if (curr == "-interval" && (++pos) < argc) {
      INTERV = lexical_cast<int>(argv[pos]);
    } else if (curr == "-vbtol" && (++pos) < argc) {
      VBTOL = lexical_cast<float>(argv[pos]);
    } else if (curr == "-jitter" && (++pos) < argc) {
      JITTER = lexical_cast<float>(argv[pos]);
    } else if (curr == "-rate0" && (++pos) < argc) {
      RATE0 = lexical_cast<float>(argv[pos]);
    } else if (curr == "-decay" && (++pos) < argc) {
      DECAY = lexical_cast<float>(argv[pos]);
    } else if (curr == "-tau_lodds" && (++pos) < argc) {
      TAU_LODDS_LB = lexical_cast<float>(argv[pos]);
      TAU_LODDS_UB = lexical_cast<float>(argv[pos]);
    } else if (curr == "-pi_lodds" && (++pos) < argc) {
      PI_LODDS_LB = lexical_cast<float>(argv[pos]);
      PI_LODDS_UB = lexical_cast<float>(argv[pos]);
    } else if (curr == "-gammax" && (++pos) < argc) {
      GAMMAX = lexical_cast<float>(argv[pos]);
    }
  }

  return EXIT_SUCCESS;
}

int main(const int argc, const char *argv[]) {

  options_t opt;

  if (argc < 1 || opt.parse_command(argc, argv) == EXIT_FAILURE) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Scalar = float;
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Index = Mat::Index;

  Mat Y;
  Mat X;
  Mat C;

  ////////////////////////////////////////////////////////////////
  // 1. read X and Y matrix
  ASSERT(read_data_file(opt.Y_FILE, Y) == EXIT_SUCCESS,
         "Failed to read Y file");
  ASSERT(read_data_file(opt.X_FILE, X) == EXIT_SUCCESS,
         "Failed to read X file");
  ASSERT(Y.rows() == X.rows(),
         "X and Y must have the same number of individuals, but Y["
             << Y.rows() << "] != X[" << X.rows() << "]");

  if (opt.COV_FILE.size() > 0) {
    ASSERT(read_data_file(opt.COV_FILE, C) == EXIT_SUCCESS,
           "Failed to read COV file");

    ASSERT(C.rows() == X.rows(),
           "X and C must have the same number of individuals, but C["
               << C.rows() << "] != X[" << X.rows() << "]");

    C.conservativeResize(Eigen::NoChange, C.cols() + 1);
    C.col(C.cols() - 1) = Mat::Ones(C.rows(), 1);
  } else {
    C.resize(X.rows(), 1);
    C.setOnes();
  }

  // construct kinship matrix to take into accounts of genetic correlation matrix
  Mat L;
  safe_chol_xxt(X, L);
  center_columns(X);
  Mat Xsq = X.cwiseProduct(X);

  TLOG("Built Cholesky for the kinship matrix XX'/p");

  ////////////////////////////////////////////////////////////////
  // 2. obtain marginal z-scores
  Mat Z(X.cols(), Y.cols());
  Mat LO(X.cols(), Y.cols());
  Mat Theta(X.cols(), Y.cols());
  Mat ThetaVar(X.cols(), Y.cols());
  Mat yy(Y.rows(), 1);
  Mat PVE(Y.cols(), 1);

  for (Index j = 0; j < Y.cols(); ++j) {
    yy = Y.col(j);
    normal_qtl_model_t<Mat> model(yy);
    auto mean_theta = make_dense_spike_slab<Scalar>(X.cols(), yy.cols(), opt);
    auto mean_eta = make_regression_eta(X, yy, mean_theta);
    auto cov_theta = make_dense_spike_slab<Scalar>(C.cols(), yy.cols(), opt);
    auto cov_eta = make_regression_eta(C, yy, cov_theta);

    auto pve = train_regression(model, mean_eta, cov_eta, L, opt);
    PVE(j) = pve(0);

    LO.col(j) = log_odds_param(mean_theta);
    Theta.col(j) = mean_param(mean_theta);
    ThetaVar.col(j) = var_param(mean_theta);

    // mu = X * E[eta] and var = X2 * V[eta]
    Z.col(j) = X.transpose() * mean_eta.repr_mean();
    Z.col(j) = Z.col(j).cwiseQuotient((Xsq.transpose() * mean_eta.repr_var()).cwiseSqrt());

    TLOG("Finished : " << (j + 1) << " / " << Y.cols() << " PVE = " << PVE(j));
  }



  ////////////////////////////////////////////////////////////////
  write_data_file(opt.OUTPUT + ".Z.txt.gz", Z);
  write_data_file(opt.OUTPUT + ".theta.txt.gz", Theta);
  write_data_file(opt.OUTPUT + ".theta_var.txt.gz", ThetaVar);
  write_data_file(opt.OUTPUT + ".lodds.txt.gz", LO);
  write_data_file(opt.OUTPUT + ".pve.txt.gz", PVE);

  TLOG("Successfully completed z-score calculation.");

  return EXIT_SUCCESS;
}
