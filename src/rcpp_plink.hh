// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include <algorithm>
#include <bitset>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "rcpp_util.hh"

#ifndef RCPP_PLINK_HH_
#define RCPP_PLINK_HH_

using BYTE = unsigned char;

///////////////////////////////////////////////////////
// modified from https://github.com/gabraham/plink2R //
///////////////////////////////////////////////////////

// For the genotype data, each byte encodes up to four genotypes (2
// bits per genoytpe). The coding is
//
//        00  Homozygote "1"/"1"
//        01  Heterozygote
//        11  Homozygote "2"/"2"
//        10  Missing genotype
//
// The only slightly confusing wrinkle is that each byte is
// effectively read backwards. That is, if we label each of the 8
// position as A to H, we would label backwards:
//
// 01101100
// HGFEDCBA
//
// and so the first four genotypes are read as follows:
// 01101100
// HGFEDCBA
//
//       AB   00  -- homozygote (first)
//     CD     11  -- other homozygote (second)
//   EF       01  -- heterozygote (third)
// GH         10  -- missing genotype (fourth)
//
// Finally, when we reach the end of a SNP (or if in
// individual-mode, the end of an individual) we skip to the start
// of a new byte (i.e. skip any remaining bits in that byte).  It is
// important to remember that the files test.bim and test.fam will
// already have been read in, so PLINK knows how many SNPs and
// individuals to expect.

auto geno_dosage(const BYTE geno);
bool check_bed_format(std::ifstream& ifs);
Rcpp::IntegerMatrix _read_plink_bed(const std::string file_name,
                                    const unsigned int N,
                                    const unsigned int NSNP);

#endif
