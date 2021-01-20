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

#ifdef GTBENCH_USE_ZLIB
#include <zlib.h>
#endif

#include <gtbench/io/base64.hpp>
#include <gtbench/io/util.hpp>
#include <gtbench/io/vtk.hpp>

namespace gtbench {
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

struct time_series {
  std::string m_filename;
  vec<std::size_t, 3> m_global_resolution, m_local_resolution, m_local_offset;
  std::vector<real_t> m_times = {};

  static void write_base64_data(std::ostream &out,
                                std::vector<real_t> const &data) {
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

  void write_data(std::ostream &out, storage_t const &data) const {
    const auto view = data->const_host_view();

    std::vector<real_t> buffer((m_local_resolution.x + 1) *
                               (m_local_resolution.y + 1) *
                               (m_local_resolution.z + 1));
#pragma omp parallel for collapse(3)
    for (std::size_t k = 0; k <= m_local_resolution.z; ++k)
      for (std::size_t j = 0; j <= m_local_resolution.y; ++j)
        for (std::size_t i = 0; i <= m_local_resolution.x; ++i)
          buffer[i + (m_local_resolution.x + 1) *
                         (j + (m_local_resolution.y + 1) * k)] =
              view(halo + i, halo + j, k);

    write_base64_data(out, buffer);
  }

  void write_velocity(std::ostream &out, storage_t const &u, storage_t const &v,
                      storage_t const &w) const {
    const auto u_view = u->const_host_view();
    const auto v_view = v->const_host_view();
    const auto w_view = w->const_host_view();

    std::vector<real_t> buffer((m_local_resolution.x + 1) *
                               (m_local_resolution.y + 1) *
                               (m_local_resolution.z + 1) * 3);
#pragma omp parallel for collapse(3)
    for (std::size_t k = 0; k <= m_local_resolution.z; ++k)
      for (std::size_t j = 0; j <= m_local_resolution.y; ++j)
        for (std::size_t i = 0; i <= m_local_resolution.x; ++i) {
          std::size_t index =
              3 * (i + (m_local_resolution.x + 1) *
                           (j + (m_local_resolution.y + 1) * k));
          buffer[index] = u_view(halo + i, halo + j, k);
          buffer[index + 1] = v_view(halo + i, halo + j, k);
          buffer[index + 2] = (w_view(halo + i, halo + j, k) +
                               w_view(halo + i, halo + j, k + 1)) /
                              2;
        }

    write_base64_data(out, buffer);
  }

  void write_pvd() const {
    std::ofstream out(m_filename);

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile"             //
        << " type=\"Collection\"" //
        << " version=\"1.0\""     //
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        << " byte_order=\"BigEndian\""
#else
        << " byte_order=\"LittleEndian\""
#endif
        << " header_type=\"UInt64\""
#ifdef GTBENCH_USE_ZLIB
        << " compressor=\"vtkZLibDataCompressor\""
#endif
        << ">\n";
    out << "<Collection>\n";

    for (std::size_t step = 0; step < m_times.size(); ++step) {
      out << "<DataSet"                                              //
          << " timestep=\"" << std::to_string(m_times[step]) << "\"" //
          << " group=\"\""                                           //
          << " part=\"0\""                                           //
          << " file=\"" << m_filename << "." << step << ".pvti\""    //
          << "/>\n";
    }

    out << "</Collection>\n";
    out << "</VTKFile>\n";
  }

  void write_pvti(numerics::solver_state const &state) const {
    std::ofstream out(m_filename + "." + std::to_string(m_times.size() - 1) +
                      ".pvti");

    std::array<std::size_t, 6> global_extent = {0, m_global_resolution.x,
                                                0, m_global_resolution.y,
                                                0, m_global_resolution.z};

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
    out << "<PPointData"           //
        << " Scalars=\"data\""     //
        << " Vectors=\"velocity\"" //
        << ">\n";
    out << "<PDataArray"                                   //
        << " Name=\"data\""                                //
        << " NumberOfComponents=\"1\" "                    //
        << " type=\"Float" << (8 * sizeof(real_t)) << "\"" //
        << "/>\n";
    out << "<PDataArray"                                   //
        << " Name=\"velocity\""                            //
        << " NumberOfComponents=\"3\" "                    //
        << " type=\"Float" << (8 * sizeof(real_t)) << "\"" //
        << "/>\n";
    out << "</PPointData>\n";
    for (std::size_t k = 0; k < m_global_resolution.z;
         k += m_local_resolution.z) {
      for (std::size_t j = 0; j < m_global_resolution.y;
           j += m_local_resolution.y) {
        for (std::size_t i = 0; i < m_global_resolution.x;
             i += m_local_resolution.x) {
          std::array<std::size_t, 6> extent = {i, i + m_local_resolution.x,
                                               j, j + m_local_resolution.y,
                                               k, k + m_local_resolution.z};
          out << "<Piece Extent=\"" << extent << "\"" //
              << " Source=\"" << m_filename << "." << (m_times.size() - 1)
              << "." << rank(m_global_resolution, m_local_resolution, {i, j, k})
              << ".vti\"" //
              << "/>\n";
        }
      }
    }
    out << " </PImageData>\n";
    out << "</VTKFile>\n";
  }

  void write_vti(numerics::solver_state const &state) const {
    std::ofstream out(m_filename + "." + std::to_string(m_times.size() - 1) +
                      "." +
                      std::to_string(rank(m_global_resolution,
                                          m_local_resolution, m_local_offset)) +
                      ".vti");

    assert(state.resolution.x == m_local_resolution.x);
    assert(state.resolution.y == m_local_resolution.y);
    assert(state.resolution.z == m_local_resolution.z);
    std::array<std::size_t, 6> offset_extent = {
        m_local_offset.x, m_local_offset.x + m_local_resolution.x,
        m_local_offset.y, m_local_offset.y + m_local_resolution.y,
        m_local_offset.z, m_local_offset.z + m_local_resolution.z,
    };

    out << "<?xml version=\"1.0\"?>\n";

    out << "<VTKFile"            //
        << " type=\"ImageData\"" //
        << " version=\"1.0\""    //
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        << " byte_order=\"BigEndian\""
#else
        << " byte_order=\"LittleEndian\""
#endif
        << " header_type=\"UInt64\""
#ifdef GTBENCH_USE_ZLIB
        << " compressor=\"vtkZLibDataCompressor\""
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

    out << "<PointData"            //
        << " Scalars=\"data\""     //
        << " Vectors=\"velocity\"" //
        << ">\n";

    out << "<DataArray"                                    //
        << " Name=\"data\""                                //
        << " NumberOfComponents=\"1\""                     //
        << " type=\"Float" << (8 * sizeof(real_t)) << "\"" //
        << " format=\"binary\""                            //
        << ">\n";

    write_data(out, state.data);

    out << "\n</DataArray>\n";

    out << "<DataArray"                                    //
        << " Name=\"velocity\""                            //
        << " NumberOfComponents=\"3\""                     //
        << " type=\"Float" << (8 * sizeof(real_t)) << "\"" //
        << " format=\"binary\""                            //
        << ">\n";

    write_velocity(out, state.u, state.v, state.w);

    out << "\n</DataArray>\n";
    out << "</PointData>\n";
    out << "</Piece>\n";
    out << "</ImageData>\n";
    out << "</VTKFile>\n";
  }
  void operator()(real_t time, numerics::solver_state const &state) {
    m_times.push_back(time);

    write_vti(state);
    if (rank(m_global_resolution, m_local_resolution, m_local_offset) == 0) {
      write_pvti(state);
      write_pvd();
    }
  }
};

} // namespace

std::function<void(real_t, numerics::solver_state const &state)>
write_time_series(std::string const &filename,
                  vec<std::size_t, 3> const &global_resolution,
                  vec<std::size_t, 3> const &local_resolution,
                  vec<std::size_t, 3> const &local_offset) {
  return time_series{filename, global_resolution, local_resolution,
                     local_offset};
}

} // namespace vtk

} // namespace io
} // namespace gtbench
