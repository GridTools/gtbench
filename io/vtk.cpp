/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>

#include "./base64.hpp"
#include "./vtk.hpp"

namespace io {
namespace vtk {

namespace {
template <class T, std::size_t N>
std::ostream &operator<<(std::ostream &out, std::array<T, N> const &array) {
  out << array[0];
  for (std::size_t i = 1; i < N; ++i)
    out << " " << array[i];
  return out;
}

template <class T>
std::ostream &operator<<(std::ostream &out, vec<T, 3> const &v) {
  out << v.x << " " << v.y << " " << v.z;
  return out;
}

void write_base64_data(std::ostream &out, std::vector<real_t> const &data) {
  std::uint64_t size = data.size() * sizeof(real_t);
#ifdef GTBENCH_USE_ZLIB
  auto compressed_size = compressBound(size);
  std::vector<Bytef> buffer(compressed_size);
  if (compress(buffer.data(), &compressed_size, (const Bytef *)data.data(),
               size) != Z_OK) {
    throw std::runtime_error("data compression failed");
  }

  std::uint64_t header[4] = {
      1,               // blocks
      size,            // uncompressed block size
      0,               // last block size
      compressed_size, // block sizes after compression
  };

  base64_encoder(out)
      .write(header, sizeof(header))
      .flush()
      .write(buffer.data(), compressed_size);
#else
  base64_encoder(out).write(&size, sizeof(size)).write(data.data(), size);
#endif
}

void write_storage_data(std::ostream &out, storage_t const &storage,
                        vec<std::size_t, 3> const &resolution) {
  const auto view = gt::make_host_view<gt::access_mode::read_only>(storage);

  std::vector<real_t> buffer((resolution.x + 1) * (resolution.y + 1) *
                             (resolution.z + 1));
#pragma omp parallel for collapse(3)
  for (std::size_t k = 0; k <= resolution.z; ++k)
    for (std::size_t j = 0; j <= resolution.y; ++j)
      for (std::size_t i = 0; i <= resolution.x; ++i)
        buffer[i + (resolution.x + 1) * (j + (resolution.y + 1) * k)] =
            view(halo + i, halo + j, k);

  write_base64_data(out, buffer);
}

} // namespace

void time_series::write_pvd() const {
  std::ofstream out(filename());

  out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile" //
      << " type=\"Collection\"" //
      << " version=\"1.0\"" //
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      << " byte_order=\"BigEndian\""
#else
      << " byte_order=\"LittleEndian\""
#endif
      << " header_type=\"UInt64\""
#ifdef GTBENCH_USE_ZLIB
      << "compressor=\"vtkZLibDataCompressor\"" : "")
#endif
      << ">\n";
    out << "<Collection>\n";

    for (std::size_t step = 0; step < m_times.size(); ++step) {
      out << "<DataSet"                                              //
          << " timestep=\"" << std::to_string(m_times[step]) << "\"" //
          << " group=\"\""                                           //
          << " part=\"0\""                                           //
          << " file=\"" << filename() << "." << step << ".pvti\""    //
          << "/>\n";
    }

    out << "</Collection>\n";
    out << "</VTKFile>\n";
}

void time_series::write_pvti(numerics::solver_state const &state) const {
  std::ofstream out(filename() + "." + std::to_string(m_times.size() - 1) +
                    ".pvti");

  std::array<std::size_t, 6> global_extent = {0, global_resolution().x,
                                              0, global_resolution().y,
                                              0, global_resolution().z};

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile"             //
      << " type=\"PImageData\"" //
      << " version=\"1.0\""     //
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      << " byte_order=\"BigEndian\""
#else
      << " byte_order=\"LittleEndian\""
#endif
      << " header_type=\"UInt64\"" //
      << ">\n";
  out << "<PImageData"                              //
      << " WholeExtent=\"" << global_extent << "\"" //
      << " Origin=\"0 0 0\""                        //
      << " Spacing=\"" << state.delta << "\""       //
      << " GhostLevel=\"0\""                        //
      << ">\n";
  out << "<PPointData>\n";
  out << "<PDataArray"                                   //
      << " Name=\"data\""                                //
      << " NumberOfComponents=\"1\" "                    //
      << " type=\"Float" << (8 * sizeof(real_t)) << "\"" //
      << "/>\n";
  out << "</PPointData>\n";
  for (std::size_t k = 0; k < global_resolution().z;
       k += local_resolution().z) {
    for (std::size_t j = 0; j < global_resolution().y;
         j += local_resolution().y) {
      for (std::size_t i = 0; i < global_resolution().x;
           i += local_resolution().x) {
        std::array<std::size_t, 6> extent = {i, i + local_resolution().x,
                                             j, j + local_resolution().y,
                                             k, k + local_resolution().z};
        out << "<Piece Extent=\"" << extent << "\"" //
            << " Source=\"" << filename() << "." << (m_times.size() - 1) << "."
            << rank({i, j, k}) << ".vti\"" //
            << "/>\n";
      }
    }
  }
  out << " </PImageData>\n";
  out << "</VTKFile>\n";
}

void time_series::write_vti(numerics::solver_state const &state) const {
  std::ofstream out(filename() + "." + std::to_string(m_times.size() - 1) +
                    "." + std::to_string(rank()) + ".vti");

  assert(state.resolution.x == local_resolution().x);
  assert(state.resolution.y == local_resolution().y);
  assert(state.resolution.z == local_resolution().z);
  std::array<std::size_t, 6> offset_extent = {
      local_offset().x, local_offset().x + local_resolution().x,
      local_offset().y, local_offset().y + local_resolution().y,
      local_offset().z, local_offset().z + local_resolution().z,
  };

  out << "<?xml version=\"1.0\"?>\n";

  out << "<VTKFile" //
      << " type=\"ImageData\"" //
      << " version=\"1.0\"" //
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      << " byte_order=\"BigEndian\""
#else
      << " byte_order=\"LittleEndian\""
#endif
      << " header_type=\"UInt64\""
#ifdef GTBENCH_USE_ZLIB
      << "compressor=\"vtkZLibDataCompressor\"" : "")
#endif
      << ">\n";

  out << "<ImageData"                               //
      << " WholeExtent=\"" << offset_extent << "\"" //
      << " Origin=\"0 0 0\""                        //
      << " Spacing=\"" << state.delta << "\""       //
      << ">\n";

  out << "<Piece"                              //
      << " Extent=\"" << offset_extent << "\"" //
      << ">\n";

  out << "<PointData>\n";

  out << "<DataArray"                                    //
      << " Name=\"data\""                                //
      << " NumberOfComponents=\"1\""                     //
      << " type=\"Float" << (8 * sizeof(real_t)) << "\"" //
      << " format=\"binary\""                            //
      << ">\n";

  write_storage_data(out, state.data, state.resolution);

  out << "\n</DataArray>\n";
  out << "</PointData>\n";
  out << "</Piece>\n";
  out << "</ImageData>\n";
  out << "</VTKFile>\n";
}

void time_series::write_step(real_t time, numerics::solver_state const &state) {
  m_times.push_back(time);

  write_vti(state);
  if (rank() == 0) {
    write_pvti(state);
    write_pvd();
  }
}
} // namespace vtk

} // namespace io