#include "rcpp_plink.hh"

// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

auto geno_dosage(const BYTE geno) {
  // 01 -> (10) -> missing
  if (geno == 1) return NA_INTEGER;
  const int a1 = static_cast<int>(!(geno & 1));   // minor allele
  const int a2 = static_cast<int>(!(geno >> 1));  // minor allele
  return a1 + a2;                                 // minor allele dosage
};

#define ASSERT_IM_RET(cond, msg)         \
  if (!(cond)) {                         \
    ELOG(msg);                           \
    return Rcpp::IntegerMatrix(0, 0, 0); \
  }

bool check_bed_format(std::ifstream& ifs) {
  ///////////////////////////////////////////////////////////
  // |-magic number--| |-mode-| |--genotype data---------| //
  //                                                       //
  // 01101100 00011011 00000001 11011100 00001111 11100111 //
  //                                                       //
  // |--genotype data-cont'd--|                            //
  //                                                       //
  // 00001111 01101011 00000001                            //
  ///////////////////////////////////////////////////////////

  auto str2byte = [](const std::string str) {
    const std::bitset<8> _bset(str);
    const unsigned long _mask = 0xFF;
    return static_cast<BYTE>(_bset.to_ulong() & _mask);
  };

  const BYTE PLINK_HEADER1 = str2byte("01101100");    // 108
  const BYTE PLINK_HEADER2 = str2byte("00011011");    // 27
  const BYTE PLINK_SNP_MAJOR = str2byte("00000001");  // 1

  ifs.seekg(0, std::ios::beg);
  BYTE b;
  if (!(ifs >> b) || b != PLINK_HEADER1) {
    ELOG("Invalid PLINK bed file : " << b << " != " << PLINK_HEADER1);
    return false;
  }

  if (!(ifs >> b) || b != PLINK_HEADER2) {
    ELOG("Invalid PLINK bed file : " << b << " != " << PLINK_HEADER2);
    return false;
  }

  if (!(ifs >> b)) {
    ELOG("Unable to read SNP or IND_MAJOR");
    return false;
  }

  const bool snp_major = (b == PLINK_SNP_MAJOR) ? true : false;

  if (!snp_major) {
    ELOG("Unable to parse IND_MAJOR mode");
    return false;
  }

  return true;
}

/////////////////////////////////
// actually read genotype data //
/////////////////////////////////

Rcpp::IntegerMatrix _read_plink_bed(const std::string file_name,
                                    const unsigned int N,
                                    const unsigned int NSNP) {
  TLOG("plink file " << file_name);

  ASSERT_IM_RET(N > 0, "Invalid number of individuals " << N);

  const unsigned int SKIP_BYTE = 3;
  const unsigned int GENO_PER_BYTE = 4;

  TLOG("Start read PLINK '" << file_name << "' (BED) into memory");
  std::ifstream ifs(file_name.c_str(), std::ios::binary);

  ifs.seekg(0, std::ios::end);
  const unsigned int FILE_LEN = ifs.tellg();
  const double byte_per_snp =
      std::ceil(static_cast<double>(N) / static_cast<double>(GENO_PER_BYTE));
  const unsigned int BYTE_PER_SNP = static_cast<unsigned int>(byte_per_snp);
  const unsigned int _nsnp = (FILE_LEN - SKIP_BYTE) / BYTE_PER_SNP;

  ASSERT_IM_RET(_nsnp == NSNP,
                "Check " << file_name << " contains different number of SNPs : "
                         << _nsnp << " but expected " << NSNP)

  ASSERT_IM_RET(FILE_LEN > 3, "Check "
                                  << file_name
                                  << ", too small file size : " << FILE_LEN);

  // check BED format
  ASSERT_IM_RET(check_bed_format(ifs), "not a proper BED file");

  ifs.seekg(SKIP_BYTE, std::ios::beg);
  std::vector<BYTE> raw_data(BYTE_PER_SNP);
  std::vector<int> data(N * NSNP);
  unsigned int pos = 0;
  const BYTE MASK = 3;  // 00000011

  Progress prog(NSNP, true);

  for (unsigned int j = 0; j < NSNP; ++j) {
    if (Progress::check_abort()) {
      break;
    }
    prog.increment();

    // 1. read
    ifs.read(reinterpret_cast<char*>(raw_data.data()),
             sizeof(BYTE) * BYTE_PER_SNP);
    // 2. parse & save
    unsigned int i = 0;
    for (auto four_geno : raw_data) {
      data[pos++] = geno_dosage(four_geno & MASK);
      if (++i >= N) break;
      data[pos++] = geno_dosage((four_geno >> 2) & MASK);
      if (++i >= N) break;
      data[pos++] = geno_dosage((four_geno >> 4) & MASK);
      if (++i >= N) break;
      data[pos++] = geno_dosage((four_geno >> 6) & MASK);
      if (++i >= N) break;
    }
  }
  ifs.close();

  TLOG("Finished reading PLINK BED'" << file_name);

  return Rcpp::IntegerMatrix(N, NSNP, data.data());

}  // END of namespace

RcppExport SEXP read_plink_bed(SEXP bed_file, SEXP n, SEXP nsnp) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter<const std::string>::type file_name(bed_file);
  Rcpp::traits::input_parameter<const unsigned int>::type N(n);
  Rcpp::traits::input_parameter<const unsigned int>::type NSNP(nsnp);
  __result = Rcpp::wrap(_read_plink_bed(file_name, N, NSNP));
  return __result;
  END_RCPP
}
