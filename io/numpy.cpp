/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>

#include "./numpy.hpp"

namespace io {
namespace numpy {

namespace {
void write_storage(std::string const &filename, storage_t const &storage) {
  std::ofstream out(filename, std::ios::binary);
  std::size_t ni = storage.total_length<0>();
  std::size_t nj = storage.total_length<1>();
  std::size_t nk = storage.total_length<2>();
  std::string header =
      "{\"descr\":\"float" + std::to_string(8 * sizeof(real_t)) +
      "\",\"fortran_order\":False,\"shape\":(" + std::to_string(ni) + "," +
      std::to_string(nj) + "," + std::to_string(nk) + ")}";
  while ((8 + header.size()) % 64 != 63)
    header += " ";
  header += "\n";
  std::uint16_t header_len = header.size();
  out.write("\x93NUMPY\x01\x00", 8);
  out.write((char const *)&header_len, sizeof(header_len));
  out << header;
  auto view = gt::make_host_view<gt::access_mode::read_only>(storage);
  for (std::size_t i = 0; i < ni; ++i)
    for (std::size_t j = 0; j < nj; ++j)
      for (std::size_t k = 0; k < nk; ++k)
        out.write((char const *)&view(i, j, k), sizeof(real_t));
}
} // namespace

void time_series::write_step(real_t time, numerics::solver_state const &state) {
  std::ostringstream fname;
  std::regex suffix_re("\\.npy$");
  std::regex_replace(std::ostreambuf_iterator<char>(fname), filename().begin(),
                     filename().end(), suffix_re, ".");
  fname << "r" << std::setprecision(std::to_string(ranks() - 1).size())
        << rank() << ".";
  fname << "t" << std::setprecision(3) << std::fixed << time;

  write_storage(fname.str() + ".data.npy", state.data);
  write_storage(fname.str() + ".u.npy", state.u);
  write_storage(fname.str() + ".v.npy", state.v);
  write_storage(fname.str() + ".w.npy", state.w);
}
} // namespace numpy
} // namespace io
