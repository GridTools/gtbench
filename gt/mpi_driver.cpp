#include <cstdlib>
#include <iostream>
#include <type_traits>

#include <mpi.h>

#include "common.hpp"

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
      MPI_Type_hvector(halo_sizes[dim][1], halo_sizes[dim][0],
                       strides[1] * sizeof(real_t), mpi_real_dtype, &plane);
      MPI_Type_hvector(halo_sizes[dim][2], 1, strides[2] * sizeof(real_t),
                       plane, &halo_dtypes[dim]);
      MPI_Type_commit(&halo_dtypes[dim]);
      MPI_Type_free(&plane);
    }
  }

  void operator()(storage_t &storage) const {
    auto ptr = storage.get_storage_ptr()->get_target_ptr();
    for (int dim = 0; dim < 2; ++dim) {
      int lower, upper;
      MPI_Cart_shift(comm_cart, dim, 1, &lower, &upper);

      MPI_Sendrecv(ptr + send_offsets[dim].first, 1, halo_dtypes[dim], lower, 0,
                   ptr + recv_offsets[dim].second, 1, halo_dtypes[dim], upper,
                   0, comm_cart, MPI_STATUS_IGNORE);

      MPI_Sendrecv(ptr + send_offsets[dim].second, 1, halo_dtypes[dim], upper,
                   1, ptr + recv_offsets[dim].first, 1, halo_dtypes[dim], lower,
                   1, comm_cart, MPI_STATUS_IGNORE);
    }
  }

  MPI_Comm comm_cart;
  std::array<MPI_Datatype, 2> halo_dtypes;
  std::array<std::pair<std::size_t, std::size_t>, 2> send_offsets;
  std::array<std::pair<std::size_t, std::size_t>, 2> recv_offsets;
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (argc != 6) {
    std::cerr << "usage: " << argv[0] << " NX NY NZ PX PY" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const std::size_t nx = std::atoll(argv[1]);
  const std::size_t ny = std::atoll(argv[2]);
  const std::size_t nz = std::atoll(argv[3]);
  const int px = std::atoi(argv[4]);
  const int py = std::atoi(argv[5]);

  std::array<int, 2> dims = {px, py}, periods = {1, 1};
  MPI_Comm comm_cart;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1,
                  &comm_cart);
  int rank, size;
  MPI_Comm_rank(comm_cart, &rank);
  MPI_Comm_size(comm_cart, &size);

  std::array<int, 2> coords;
  MPI_Cart_coords(comm_cart, rank, 2, coords.data());

  std::size_t begin_x = nx * coords[0] / px;
  std::size_t end_x = nx * (coords[0] + 1) / px;
  std::size_t begin_y = ny * coords[1] / py;
  std::size_t end_y = ny * (coords[1] + 1) / py;

  std::size_t resolution_x = end_x - begin_x;
  std::size_t resolution_y = end_y - begin_y;
  std::size_t resolution_z = nz;

  if (resolution_x < 3 || resolution_y < 3) {
    std::cerr << "too few grid points per process" << std::endl;
    MPI_Abort(comm_cart, 2);
  }

  halos_t halos{
      {{halo, halo, halo, halo + gt::uint_t(resolution_x) - 1,
        halo + gt::uint_t(resolution_x) + halo},
       {halo, halo, halo, halo + gt::uint_t(resolution_y) - 1,
        halo + gt::uint_t(resolution_y) + halo},
       {0, 0, 0, gt::uint_t(resolution_z) - 1, gt::uint_t(resolution_z)}}};

  storage_t::storage_info_t sinfo(resolution_x + 2 * halo,
                                  resolution_y + 2 * halo, resolution_z);

  halo_exchange_f exchange{comm_cart, sinfo};

  storage_t storage(sinfo, rank, "storage");

  exchange(storage);

  auto view = gt::make_host_view(storage);

  for (int r = 0; r < size; ++r) {
    if (r == rank) {
      std::cout << "rank " << rank << std::endl;
      for (std::size_t k = 0; k < resolution_z; ++k) {
        for (std::size_t j = 0; j < resolution_y + 2 * halo; ++j) {
          for (std::size_t i = 0; i < resolution_x + 2 * halo; ++i)
            std::cout << view(i, j, k) << " ";
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
      std::cout.flush();
    }
    MPI_Barrier(comm_cart);
  }

  MPI_Finalize();
}