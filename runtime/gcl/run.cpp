/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./run.hpp"

#include <gridtools/communication/GCL.hpp>
#include <gridtools/communication/halo_exchange.hpp>

#include <mpi.h>

namespace runtime {
namespace gcl_impl {

using pattern_t = gt::halo_exchange_dynamic_ut<storage_info_ijk_t::layout_t,
                                               gt::layout_map<0, 1, 2>, real_t,
#ifdef __CUDACC__
                                               gt::gcl_gpu
#else
                                               gt::gcl_cpu
#endif
                                               >;

runtime::runtime(std::array<int, 2> const &cart_dims)
    : m_scope((void (*)())gt::GCL_Init, gt::GCL_Finalize),
      m_cart_dims(cart_dims) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Dims_create(size, 2, m_cart_dims.data());
  if (m_cart_dims[0] * m_cart_dims[1] != size) {
    throw std::runtime_error(
        "the product of cart dims must be equal to the number of MPI ranks.");
  }

  if (size > 1 && rank != 0)
    std::cout.setstate(std::ios_base::failbit);
}

void runtime_register_options(gcl, options &options) {
  options("cart-dims", "dimensons of cartesian communicator", "PX PY", 2);
}

runtime runtime_init(gcl, options_values const &options) {
  std::array<int, 2> cart_dims = {0, 0};
  if (options.has("cart-dims"))
    cart_dims = options.get<std::array<int, 2>>("cart-dims");

  return std::move(runtime(cart_dims));
}

struct process_grid::impl {
  impl(vec<std::size_t, 3> const &global_resolution,
       std::array<int, 2> cart_dims)
      : m_comm_cart(MPI_COMM_NULL) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::array<int, 3> cart_dims3 = {cart_dims[0], cart_dims[1], 1};
    std::array<int, 3> periods3 = {1, 1, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, cart_dims3.data(), periods3.data(), 1,
                    &m_comm_cart);
    std::array<int, 3> coords;
    MPI_Cart_coords(m_comm_cart, rank, 3, coords.data());

    m_local_offset.x = global_resolution.x * coords[0] / cart_dims[0];
    m_local_offset.y = global_resolution.y * coords[1] / cart_dims[1];
    m_local_offset.z = 0;

    vec<std::size_t, 2> next_offset = {
        global_resolution.x * (coords[0] + 1) / cart_dims[0],
        global_resolution.y * (coords[1] + 1) / cart_dims[1]};

    m_local_resolution.x = next_offset.x - m_local_offset.x;
    m_local_resolution.y = next_offset.y - m_local_offset.y;
    m_local_resolution.z = global_resolution.z;

    if (m_local_resolution.x < halo || m_local_resolution.y < halo)
      throw std::runtime_error(
          "local local_resolution is smaller than halo size!");
  }

  ~impl() { MPI_Comm_free(&m_comm_cart); }

  impl(impl const &) = delete;
  impl &operator=(impl const &) = delete;

  std::function<void(storage_t &)>
  exchanger(storage_info_ijk_t const &sinfo) const {
    auto pattern = std::make_shared<pattern_t>(
        pattern_t::grid_type::period_type{true, true, false}, m_comm_cart);

    pattern->add_halo<0>(halo, halo, halo, m_local_resolution.x + halo - 1,
                         sinfo.padded_length<0>());
    pattern->add_halo<1>(halo, halo, halo, m_local_resolution.y + halo - 1,
                         sinfo.padded_length<1>());
    pattern->add_halo<2>(0, 0, 0, m_local_resolution.z - 1,
                         sinfo.padded_length<2>());

    pattern->setup(1);

#ifdef __CUDACC__
    cudaStreamSynchronize(0);
#endif

    return [pattern = std::move(pattern)](storage_t &storage) {
      auto ptr = storage.get_storage_ptr()->get_target_ptr();
      pattern->pack(ptr);
      pattern->exchange();
      pattern->unpack(ptr);
    };
  }

  result collect_results(result r) const {
    result reduced;
    MPI_Allreduce(&r, &reduced, 2, MPI_DOUBLE, MPI_MAX, m_comm_cart);
    return reduced;
  }

  vec<std::size_t, 3> m_local_resolution, m_local_offset;
  MPI_Comm m_comm_cart;
};

process_grid::process_grid(vec<std::size_t, 3> const &global_resolution,
                           std::array<int, 2> cart_dims)
    : m_impl(std::make_unique<impl>(global_resolution, cart_dims)) {}

process_grid::~process_grid() {}

vec<std::size_t, 3> process_grid::local_resolution() const {
  return m_impl->m_local_resolution;
}
vec<std::size_t, 3> process_grid::local_offset() const {
  return m_impl->m_local_offset;
}

std::function<void(storage_t &)>
process_grid::exchanger(storage_info_ijk_t const &sinfo) const {
  return m_impl->exchanger(sinfo);
}

double process_grid::wtime() const { return MPI_Wtime(); }

result process_grid::collect_results(result r) const {
  return m_impl->collect_results(r);
}

} // namespace gcl_impl
} // namespace runtime