#include "simple_mpi.hpp"

#include <array>
#include <iostream>

#include <mpi.h>

namespace communication {

namespace simple_mpi {

world::world(int &argc, char **&argv) {
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size > 1 && rank != 0)
    std::cout.setstate(std::ios_base::failbit);
}

world::~world() { MPI_Finalize(); }

grid::grid(std::size_t global_resolution_x, std::size_t global_resolution_y,
           std::size_t resolution_z)
    : global_resolution_x(global_resolution_x),
      global_resolution_y(global_resolution_y), resolution_z(resolution_z) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::array<int, 2> dims = {0, 0}, periods = {1, 1};
  MPI_Dims_create(size, 2, dims.data());
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1,
                  &comm_cart);
  std::array<int, 2> coords;
  MPI_Cart_coords(comm_cart, rank, 2, coords.data());

  offset_x = global_resolution_x * coords[0] / dims[0];
  offset_y = global_resolution_y * coords[1] / dims[1];

  auto next_offset_x = global_resolution_x * (coords[0] + 1) / dims[0];
  auto next_offset_y = global_resolution_y * (coords[1] + 1) / dims[1];

  resolution_x = next_offset_x - offset_x;
  resolution_y = next_offset_y - offset_y;

  if (resolution_x < halo || resolution_y < halo)
    throw std::runtime_error("local resolution is smaller than halo size!");
} // namespace mpi

grid::~grid() { MPI_Comm_free(&comm_cart); }

struct halo_exchange_f {
  halo_exchange_f(MPI_Comm comm_cart, storage_t::storage_info_t const &sinfo)
      : comm_cart(comm_cart), send_offsets{{{0, 0}, {0, 0}}},
        recv_offsets{{{0, 0}, {0, 0}}} {
    auto strides = sinfo.strides();
    auto sizes = sinfo.total_lengths();

    gt::array<decltype(sizes), 2> halo_sizes;
    for (int dim = 0; dim < 2; ++dim) {
      halo_sizes[dim] = sizes;
      for (int i = 0; i < 2; ++i)
        halo_sizes[dim][i] = i == dim ? halo : sizes[i] - 2 * halo;

      for (int i = 0; i < 2; ++i) {
        if (i == dim) {
          send_offsets[dim].first += halo * strides[i];
          send_offsets[dim].second += (sizes[i] - 2 * halo) * strides[i];
          recv_offsets[dim].first += 0 * strides[i];
          recv_offsets[dim].second += (sizes[i] - halo) * strides[i];
        } else {
          send_offsets[dim].first += halo * strides[i];
          send_offsets[dim].second += halo * strides[i];
          recv_offsets[dim].first += halo * strides[i];
          recv_offsets[dim].second += halo * strides[i];
        }
      }
    }

    // sort strides and halo_sizes by strides
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j > 0 && strides[j - 1] > strides[j]; --j) {
        std::swap(strides[j], strides[j - 1]);
        for (int dim = 0; dim < 2; ++dim)
          std::swap(halo_sizes[dim][j], halo_sizes[dim][j - 1]);
      }
    }

    MPI_Datatype mpi_real_dtype;
    MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(real_t), &mpi_real_dtype);

    for (int dim = 0; dim < 2; ++dim) {
      MPI_Datatype plane;
      halo_dtypes[dim] = std::shared_ptr<MPI_Datatype>(new MPI_Datatype,
                                                       [](MPI_Datatype *type) {
                                                         MPI_Type_free(type);
                                                         delete type;
                                                       });
      MPI_Type_hvector(halo_sizes[dim][1], halo_sizes[dim][0],
                       strides[1] * sizeof(real_t), mpi_real_dtype, &plane);
      MPI_Type_hvector(halo_sizes[dim][2], 1, strides[2] * sizeof(real_t),
                       plane, halo_dtypes[dim].get());
      MPI_Type_commit(halo_dtypes[dim].get());
      MPI_Type_free(&plane);
    }
  }

  void operator()(storage_t &storage) const {
    real_t *ptr = storage.get_storage_ptr()->get_target_ptr();
    int tag = 0;
    for (int dim = 0; dim < 2; ++dim) {
      int lower, upper;
      MPI_Cart_shift(comm_cart, dim, 1, &lower, &upper);

      MPI_Sendrecv(ptr + send_offsets[dim].first, 1, *halo_dtypes[dim], lower,
                   tag, ptr + recv_offsets[dim].second, 1, *halo_dtypes[dim],
                   upper, tag, comm_cart, MPI_STATUS_IGNORE);
      ++tag;

      MPI_Sendrecv(ptr + send_offsets[dim].second, 1, *halo_dtypes[dim], upper,
                   tag, ptr + recv_offsets[dim].first, 1, *halo_dtypes[dim],
                   lower, tag, comm_cart, MPI_STATUS_IGNORE);
      ++tag;
    }
  }

  MPI_Comm comm_cart;
  std::array<std::shared_ptr<MPI_Datatype>, 2> halo_dtypes;
  std::array<std::pair<std::size_t, std::size_t>, 2> send_offsets;
  std::array<std::pair<std::size_t, std::size_t>, 2> recv_offsets;
};

std::function<void(storage_t &)>
comm_halo_exchanger(grid const &comm_grid,
                    storage_t::storage_info_t const &sinfo) {
  return halo_exchange_f(comm_grid.comm_cart, sinfo);
}

double comm_global_sum(grid const &grid, double t) {
  MPI_Allreduce(&t, &t, 1, MPI_DOUBLE, MPI_SUM, grid.comm_cart);
  return t;
}

} // namespace simple_mpi

} // namespace communication