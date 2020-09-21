/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>

#include "./numpy.hpp"
#include "./util.hpp"

namespace io {
namespace numpy {

namespace {
void write_storage(std::string const &filename, storage_t const &storage) {
  std::ofstream out(filename, std::ios::binary);

  // write numpy array format as documented here:
  // https://numpy.org/doc/1.18/reference/generated/numpy.lib.format.html#format-version-1-0

  // numpy format magic string
  const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  // numpy format version, currently 1.0 for max compatibility
  const char version[2] = {1, 0};

  // numpy array format description
  const std::size_t ni = storage->lengths()[0];
  const std::size_t nj = storage->lengths()[1];
  const std::size_t nk = storage->lengths()[2];
  std::string descr =
      "{\"descr\":\"float" + std::to_string(8 * sizeof(real_t)) +
      "\",\"fortran_order\":False,\"shape\":(" + std::to_string(ni) + "," +
      std::to_string(nj) + "," + std::to_string(nk) + ")}\n";

  // size of magic string, version number and 16bit header length
  constexpr std::uint16_t base_header_len =
      sizeof(magic) + sizeof(version) + sizeof(std::uint16_t);
  // total header length, rounded up to a multiple of 64 for alignment
  std::uint16_t total_header_len = (base_header_len + descr.size() + 63) & ~63;

  // header length as stored in the file (size of descr and padding only)
  std::uint16_t header_len = total_header_len - base_header_len;

  // write header
  out.write(magic, sizeof(magic));
  out.write(version, sizeof(version));
  out.write((const char *)&header_len, sizeof(header_len));
  out << descr;
  // padding
  while (out.tellp() < total_header_len)
    out << '\x20';
  assert(out.tellp() % 64 == 0);

  // write data in C-order
  auto view = storage->const_host_view();
  for (std::size_t i = 0; i < ni; ++i)
    for (std::size_t j = 0; j < nj; ++j)
      for (std::size_t k = 0; k < nk; ++k)
        out.write((char const *)&view(i, j, k), sizeof(real_t));
}
} // namespace

std::function<void(real_t, numerics::solver_state const &state)>
write_time_series(std::string const &filename,
                  vec<std::size_t, 3> const &global_resolution,
                  vec<std::size_t, 3> const &local_resolution,
                  vec<std::size_t, 3> const &local_offset) {
  auto ranks = io::ranks(global_resolution, local_resolution);
  auto rank = io::rank(global_resolution, local_resolution, local_offset);
  return [=](real_t time, numerics::solver_state const &state) {
    std::ostringstream fname;
    std::regex suffix_re("\\.npy$");
    std::regex_replace(std::ostreambuf_iterator<char>(fname), filename.begin(),
                       filename.end(), suffix_re, ".");
    fname << "r" << std::setprecision(std::to_string(ranks - 1).size()) << rank
          << ".";
    fname << "t" << std::setprecision(3) << std::fixed << time;

    write_storage(fname.str() + ".data.npy", state.data);
    write_storage(fname.str() + ".u.npy", state.u);
    write_storage(fname.str() + ".v.npy", state.v);
    write_storage(fname.str() + ".w.npy", state.w);
  };
}

} // namespace numpy
} // namespace io
