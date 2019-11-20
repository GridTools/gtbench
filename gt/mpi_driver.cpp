#include <cstdlib>
#include <iostream>
#include <type_traits>

#include <mpi.h>

#include "common.hpp"

MPI_Datatype cube_dtype(gt::array<gt::uint_t, 3> strides,
                        gt::array<gt::uint_t, 3> sizes) {
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j > 0 && strides[j - 1] > strides[j]; --j) {
      std::swap(strides[j], strides[j - 1]);
      std::swap(sizes[j], sizes[j - 1]);
    }
  }

  auto mpi_type = std::is_same<real_t, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  MPI_Datatype plane;
  MPI_Type_hvector(sizes[1], sizes[0], strides[1] * sizeof(real_t), mpi_type,
                   &plane);
  MPI_Type_commit(&plane);
  MPI_Datatype cube;
  MPI_Type_hvector(sizes[2], 1, strides[2] * sizeof(real_t), plane, &cube);
  MPI_Type_commit(&cube);
  return cube;
}

struct halo_exchange_f {
  void operator()(storage_t &storage) const {
    real_t *ptr = storage.get_storage_ptr()->get_target_ptr();
    auto strides = storage.strides();
    gt::uint_t size_x = storage.total_length<0>();
    gt::uint_t size_y = storage.total_length<1>();
    gt::uint_t size_z = storage.total_length<2>();

    auto halo_x = cube_dtype(strides, {halo, size_y - 2 * halo, size_z});
    auto halo_y = cube_dtype(strides, {size_x - 2 * halo, halo, size_z});

    auto start = [&](gt::uint_t i, gt::uint_t j) {
      return ptr + i * strides[0] + j * strides[1];
    };

    int lower_x, upper_x;
    MPI_Cart_shift(comm_cart, 0, 1, &lower_x, &upper_x);
    int lower_y, upper_y;
    MPI_Cart_shift(comm_cart, 1, 1, &lower_y, &upper_y);

    MPI_Sendrecv(start(halo, halo), 1, halo_x, lower_x, 0,
                 start(size_x - halo, halo), 1, halo_x, upper_x, 0, comm_cart,
                 MPI_STATUS_IGNORE);

    MPI_Sendrecv(start(size_x - 2 * halo, halo), 1, halo_x, upper_x, 1,
                 start(0, halo), 1, halo_x, lower_x, 1, comm_cart,
                 MPI_STATUS_IGNORE);

    MPI_Sendrecv(start(halo, halo), 1, halo_y, lower_y, 2,
                 start(halo, size_y - halo), 1, halo_y, upper_y, 2, comm_cart,
                 MPI_STATUS_IGNORE);

    MPI_Sendrecv(start(halo, size_y - 2 * halo), 1, halo_y, upper_y, 3,
                 start(halo, 0), 1, halo_y, lower_y, 3, comm_cart,
                 MPI_STATUS_IGNORE);
  }

  MPI_Comm comm_cart;
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

  halo_exchange_f exchange{comm_cart};

  storage_t::storage_info_t sinfo(resolution_x + 2 * halo,
                                  resolution_y + 2 * halo, resolution_z);

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