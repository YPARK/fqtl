#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "gzstream.hh"
#include "strbuf.hh"
#include "util.hh"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifndef UTIL_IO_HH_
#define UTIL_IO_HH_

////////////////////////////////////////////////////////////////
// I/O routines

auto is_gz(const std::string filename) {
  if (filename.size() < 3) return false;
  return filename.substr(filename.size() - 3) == ".gz";
}

std::shared_ptr<std::ifstream> open_ifstream(const std::string filename) {
  std::shared_ptr<std::ifstream> ret(new std::ifstream(filename.c_str(), std::ios::in));
  return ret;
}

std::shared_ptr<igzstream> open_igzstream(const std::string filename) {
  std::shared_ptr<igzstream> ret(new igzstream(filename.c_str(), std::ios::in));
  return ret;
}

template <typename IFS, typename T>
auto read_vector_stream(IFS &ifs, std::vector<T> &in) {
  in.clear();
  T v;
  while (ifs >> v) {
    in.push_back(v);
  }
  ERR_RET(in.size() == 0, "empty vector");
  return EXIT_SUCCESS;
}

template <typename T>
auto read_vector_file(const std::string filename, std::vector<T> &in) {
  auto ret = EXIT_SUCCESS;

  if (is_gz(filename)) {
    igzstream ifs(filename.c_str(), std::ios::in);
    ret = read_vector_stream(ifs, in);
    ifs.close();
  } else {
    std::ifstream ifs(filename.c_str(), std::ios::in);
    ret = read_vector_stream(ifs, in);
    ifs.close();
  }
  return ret;
}

template <typename IFS>
auto num_cols(IFS &ifs) {
  std::istreambuf_iterator<char> eos;
  std::istreambuf_iterator<char> it(ifs);
  const auto eol = '\n';

  auto ret = 1;
  for (; it != eos && *it != eol; ++it) {
    char c = *it;
    if (isspace(c) && c != eol) ++ret;
  }

  return ret;
}

template <typename IFS>
auto num_rows(IFS &ifs) {
  std::istreambuf_iterator<char> eos;
  std::istreambuf_iterator<char> it(ifs);
  const char eol = '\n';

  auto ret = 0;
  for (; it != eos; ++it)
    if (*it == eol) ++ret;

  return ret;
}

template <typename IFS, typename T>
auto read_data_stream(IFS &ifs, T &in) {
  typedef typename T::Scalar elem_t;

  typedef enum _state_t { S_WORD, S_EOW, S_EOL } state_t;
  const auto eol = '\n';
  std::istreambuf_iterator<char> END;
  std::istreambuf_iterator<char> it(ifs);

  std::vector<elem_t> data;
  strbuf_t strbuf;
  state_t state = S_EOL;

  auto nr = 0u;  // number of rows
  auto nc = 1u;  // number of columns

  elem_t val;
  auto nmissing = 0;

  for (; it != END; ++it) {
    char c = *it;

    if (c == eol) {
      if (state == S_WORD) {
        val = strbuf.lexical_cast<elem_t>();

        if (!isfinite(val)) nmissing++;

        data.push_back(val);
        strbuf.clear();
      } else if (state == S_EOW) {
        data.push_back(NAN);
        nmissing++;
      }
      state = S_EOL;
      nr++;
    } else if (isspace(c)) {
      if (state == S_WORD) {
        val = strbuf.lexical_cast<elem_t>();

        if (!isfinite(val)) nmissing++;

        data.push_back(val);
        strbuf.clear();
      } else {
        data.push_back(NAN);
        nmissing++;
      }
      state = S_EOW;
      if (nr == 0) nc++;

    } else {
      strbuf.add(c);
      state = S_WORD;
    }
  }

#ifdef DEBUG
  TLOG("Found " << nmissing << " missing values");
#endif

  auto mtot = data.size();
  ERR_RET(mtot != (nr * nc), "# data points: " << mtot << " elements in " << nr << " x " << nc << " matrix");
  ERR_RET(mtot < 1, "empty file");
  ERR_RET(nr < 1, "zero number of rows; incomplete line?");
  in = Eigen::Map<T>(data.data(), nc, nr);
  in.transposeInPlace();

  return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////
template <typename T>
auto read_data_file(const std::string filename, T &in) {
  auto ret = EXIT_SUCCESS;

  if (is_gz(filename)) {
    igzstream ifs(filename.c_str(), std::ios::in);
    ret = read_data_stream(ifs, in);
    ifs.close();
  } else {
    std::ifstream ifs(filename.c_str(), std::ios::in);
    ret = read_data_stream(ifs, in);
    ifs.close();
  }

  return ret;
}

////////////////////////////////////////////////////////////////
template <typename T>
auto read_data_file(const std::string filename) {
  typename std::shared_ptr<T> ret(new T{});
  auto &in = *ret.get();

  if (is_gz(filename)) {
    igzstream ifs(filename.c_str(), std::ios::in);
    CHK_ERR_EXIT(read_data_stream(ifs, in), "Failed to read " << filename);
    ifs.close();
  } else {
    std::ifstream ifs(filename.c_str(), std::ios::in);
    CHK_ERR_EXIT(read_data_stream(ifs, in), "Failed to read " << filename);
    ifs.close();
  }

  return ret;
}

////////////////////////////////////////////////////////////////
template <typename OFS, typename Derived>
void write_data_stream(OFS &ofs, const Eigen::MatrixBase<Derived> &out) {
  ofs.precision(4);

  const Derived &M = out.derived();

  for (auto r = 0u; r < M.rows(); ++r) {
    ofs << M.coeff(r, 0);
    for (auto c = 1u; c < M.cols(); ++c) ofs << " " << M.coeff(r, c);
    ofs << std::endl;
  }
}

template <typename OFS, typename Derived>
void write_data_stream(OFS &ofs, const Eigen::SparseMatrixBase<Derived> &out) {
  ofs.precision(4);

  const Derived &M = out.derived();

  // column major
  for (auto k = 0; k < M.outerSize(); ++k) {
    for (typename Derived::InnerIterator it(M, k); it; ++it) {
      const auto i = it.row();
      const auto j = it.col();
      const auto v = it.value();
      ofs << i << " " << j << " " << v << std::endl;
    }
  }
}

////////////////////////////////////////////////////////////////
template <typename T>
void write_data_file(const std::string filename, const T &out) {
  if (is_gz(filename)) {
    ogzstream ofs(filename.c_str(), std::ios::out);
    write_data_stream(ofs, out);
    ofs.close();
  } else {
    std::ofstream ofs(filename.c_str(), std::ios::out);
    write_data_stream(ofs, out);
    ofs.close();
  }
}

#endif
