/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <mpi.h>

#include "./run.hpp"

namespace runtime {
namespace simple_mpi {

world::world(int &argc, char **&argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size > 1 && rank != 0)
    std::cout.setstate(std::ios_base::failbit);
}

world::~world() { MPI_Finalize(); }

void runtime_register_options(world const &, options &options) {
  options("cart-dims", "dimensons of cartesian communicator", "PX PY", 2);
}

runtime runtime_init(world const &, options_values const &options) {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  runtime rt = {0};
  if (options.has("cart-dims")) {
    auto values = options.get<std::array<int, 2>>("cart-dims");
    if (values.size() != 2) {
      throw std::runtime_error("wrong number of arguments in --cart-dims.");
    }
    if (values[0] * values[1] != size) {
      throw std::runtime_error(
          "the product of cart dims must be equal to the number of MPI ranks.");
    }
    std::copy(std::begin(values), std::end(values), std::begin(rt.cart_dims));
  } else {
    MPI_Dims_create(size, 2, rt.cart_dims.data());
  }

  return rt;
}

template <class T> struct halo_info { T lower, upper; };

struct process_grid::impl {
  impl(vec<std::size_t, 3> const &global_resolution,
       std::array<int, 2> cart_dims)
      : comm_cart(MPI_COMM_NULL) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::array<int, 2> periods = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, cart_dims.data(), periods.data(), 1,
                    &comm_cart);
    std::array<int, 2> coords;
    MPI_Cart_coords(comm_cart, rank, 2, coords.data());

    local_offset.x = global_resolution.x * coords[0] / cart_dims[0];
    local_offset.y = global_resolution.y * coords[1] / cart_dims[1];

    vec<std::size_t, 2> next_offset = {
        global_resolution.x * (coords[0] + 1) / cart_dims[0],
        global_resolution.y * (coords[1] + 1) / cart_dims[1]};

    local_resolution.x = next_offset.x - local_offset.x;
    local_resolution.y = next_offset.y - local_offset.y;
    local_resolution.z = global_resolution.z;

    if (local_resolution.x < halo || local_resolution.y < halo)
      throw std::runtime_error(
          "local local_resolution is smaller than halo size!");
  }

  ~impl() {
    if (comm_cart != MPI_COMM_NULL)
      MPI_Comm_free(&comm_cart);
  }

  std::function<void(storage_t &)>
  exchanger(storage_info_ijk_t const &sinfo) const {
    auto strides = sinfo.strides();
    auto sizes = sinfo.total_lengths();

    // sized of halos to exchange along x- and y-axes
    vec<decltype(sizes), 2> halo_sizes;
    halo_sizes.x = {halo, sizes[1] - 2 * halo, sizes[2]};
    halo_sizes.y = {sizes[0] - 2 * halo, halo, sizes[2]};

    // send and recv offsets for upper and lower halos
    vec<halo_info<std::size_t>, 2> send_offsets, recv_offsets;

    // send- and recv-offsets along x-axis
    send_offsets.x.lower = halo * strides[0] + halo * strides[1];
    recv_offsets.x.lower = halo * strides[1];
    send_offsets.x.upper =
        (sizes[0] - 2 * halo) * strides[0] + halo * strides[1];
    recv_offsets.x.upper = (sizes[0] - halo) * strides[0] + halo * strides[1];

    // send- and recv-offsets along y-axis
    send_offsets.y.lower = halo * strides[0] + halo * strides[1];
    recv_offsets.y.lower = halo * strides[0];
    send_offsets.y.upper =
        halo * strides[0] + (sizes[1] - 2 * halo) * strides[1];
    recv_offsets.y.upper = halo * strides[0] + (sizes[1] - halo) * strides[1];

    // sort strides and halo_sizes by strides to simplify building of the MPI
    // data types
    for (int i = 1; i < 3; ++i)
      for (int j = i; j > 0 && strides[j - 1] > strides[j]; --j) {
        std::swap(strides[j], strides[j - 1]);
        std::swap(halo_sizes.x[j], halo_sizes.x[j - 1]);
        std::swap(halo_sizes.y[j], halo_sizes.y[j - 1]);
      }

    MPI_Datatype mpi_real_dtype, plane;
    MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(real_t), &mpi_real_dtype);

    auto mpi_dtype_deleter = [](MPI_Datatype *type) {
      MPI_Type_free(type);
      delete type;
    };

    // MPI dtypes for halos along x- and y-axes (shared ptr to make the functor
    // copyable and thus usable inside std::function<>)
    vec<std::shared_ptr<MPI_Datatype>, 2> halo_dtypes;

    // building of halo data type for exchanges along x-axis
    halo_dtypes.x =
        std::shared_ptr<MPI_Datatype>(new MPI_Datatype, mpi_dtype_deleter);
    MPI_Type_create_hvector(halo_sizes.x[1], halo_sizes.x[0],
                            strides[1] * sizeof(real_t), mpi_real_dtype,
                            &plane);
    MPI_Type_create_hvector(halo_sizes.x[2], 1, strides[2] * sizeof(real_t),
                            plane, halo_dtypes.x.get());
    MPI_Type_commit(halo_dtypes.x.get());
    MPI_Type_free(&plane);

    // building of halo data type for exchanges along y-axis
    halo_dtypes.y =
        std::shared_ptr<MPI_Datatype>(new MPI_Datatype, mpi_dtype_deleter);
    MPI_Type_create_hvector(halo_sizes.y[1], halo_sizes.y[0],
                            strides[1] * sizeof(real_t), mpi_real_dtype,
                            &plane);
    MPI_Type_create_hvector(halo_sizes.y[2], 1, strides[2] * sizeof(real_t),
                            plane, halo_dtypes.y.get());
    MPI_Type_commit(halo_dtypes.y.get());
    MPI_Type_free(&plane);

    return [this, halo_dtypes = std::move(halo_dtypes),
            recv_offsets = std::move(recv_offsets),
            send_offsets = std::move(send_offsets)](storage_t &storage) {
      // storage data pointer
      real_t *ptr = storage.get_storage_ptr()->get_target_ptr();

      // neighbor ranks along x- and y-axes
      vec<halo_info<int>, 2> nb;
      MPI_Cart_shift(comm_cart, 0, 1, &nb.x.lower, &nb.x.upper);
      MPI_Cart_shift(comm_cart, 1, 1, &nb.y.lower, &nb.y.upper);

      // halo exchange along x-axis
      MPI_Sendrecv(ptr + send_offsets.x.lower, 1, *halo_dtypes.x, nb.x.lower, 0,
                   ptr + recv_offsets.x.upper, 1, *halo_dtypes.x, nb.x.upper, 0,
                   comm_cart, MPI_STATUS_IGNORE);
      MPI_Sendrecv(ptr + send_offsets.x.upper, 1, *halo_dtypes.x, nb.x.upper, 1,
                   ptr + recv_offsets.x.lower, 1, *halo_dtypes.x, nb.x.lower, 1,
                   comm_cart, MPI_STATUS_IGNORE);
      // halo exchange along y-axis
      MPI_Sendrecv(ptr + send_offsets.y.lower, 1, *halo_dtypes.y, nb.y.lower, 2,
                   ptr + recv_offsets.y.upper, 1, *halo_dtypes.y, nb.y.upper, 2,
                   comm_cart, MPI_STATUS_IGNORE);
      MPI_Sendrecv(ptr + send_offsets.y.upper, 1, *halo_dtypes.y, nb.y.upper, 3,
                   ptr + recv_offsets.y.lower, 1, *halo_dtypes.y, nb.y.lower, 3,
                   comm_cart, MPI_STATUS_IGNORE);
    };
  }

  result collect_results(result r) const {
    result reduced;
    MPI_Allreduce(&r, &reduced, 2, MPI_DOUBLE, MPI_MAX, comm_cart);
    return reduced;
  }

  vec<std::size_t, 3> local_resolution;
  vec<std::size_t, 2> local_offset;
  // cartesian communicator
  MPI_Comm comm_cart;
};

process_grid::process_grid(vec<std::size_t, 3> const &global_resolution,
                           std::array<int, 2> cart_dims)
    : pimpl(std::make_unique<impl>(global_resolution, cart_dims)) {}

process_grid::~process_grid() {}

vec<std::size_t, 3> process_grid::local_resolution() const {
  return pimpl->local_resolution;
}
vec<std::size_t, 2> process_grid::local_offset() const {
  return pimpl->local_offset;
}

std::function<void(storage_t &)>
process_grid::exchanger(storage_info_ijk_t const &sinfo) const {
  return pimpl->exchanger(sinfo);
}

double process_grid::wtime() const { return MPI_Wtime(); }

result process_grid::collect_results(result r) const {
  return pimpl->collect_results(r);
}

} // namespace simple_mpi
} // namespace runtime