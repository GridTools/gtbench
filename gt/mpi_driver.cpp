#include <cstdlib>
#include <iostream>
#include <type_traits>

#include <mpi.h>

#include "common.hpp"
#include "solver_state.hpp"
#include "verification/analytical.hpp"

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
    int tag = 0;
    for (int dim = 0; dim < 2; ++dim) {
      int lower, upper;
      MPI_Cart_shift(comm_cart, dim, 1, &lower, &upper);

      MPI_Sendrecv(ptr + send_offsets[dim].first, 1, halo_dtypes[dim], lower,
                   tag, ptr + recv_offsets[dim].second, 1, halo_dtypes[dim],
                   upper, tag, comm_cart, MPI_STATUS_IGNORE);
      ++tag;

      MPI_Sendrecv(ptr + send_offsets[dim].second, 1, halo_dtypes[dim], upper,
                   tag, ptr + recv_offsets[dim].first, 1, halo_dtypes[dim],
                   lower, tag, comm_cart, MPI_STATUS_IGNORE);
      ++tag;
    }
  }

  MPI_Comm comm_cart;
  std::array<MPI_Datatype, 2> halo_dtypes;
  std::array<std::pair<std::size_t, std::size_t>, 2> send_offsets;
  std::array<std::pair<std::size_t, std::size_t>, 2> recv_offsets;
};

struct mpi_setup {
  mpi_setup(std::size_t global_resolution_x, std::size_t global_resolution_y,
            std::size_t global_resolution_z, int procs_x, int procs_y)
      : global_resolution_x(global_resolution_x),
        global_resolution_y(global_resolution_y),
        global_resolution_z(global_resolution_z) {
    std::array<int, 2> dims = {procs_x, procs_y}, periods = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1,
                    &comm_cart);

    int rank, size;
    MPI_Comm_rank(comm_cart, &rank);
    MPI_Comm_size(comm_cart, &size);

    std::array<int, 2> coords;
    MPI_Cart_coords(comm_cart, rank, 2, coords.data());

    offset_x = global_resolution_x * coords[0] / procs_x;
    auto next_offset_x = global_resolution_x * (coords[0] + 1) / procs_x;
    offset_y = global_resolution_y * coords[1] / procs_y;
    auto next_offset_y = global_resolution_y * (coords[1] + 1) / procs_y;
    offset_z = 0;

    resolution_x = next_offset_x - offset_x;
    resolution_y = next_offset_y - offset_y;
    resolution_z = global_resolution_z;

    if (resolution_x < halo || resolution_y < halo) {
      std::cerr << "too few grid points per process" << std::endl;
      MPI_Abort(comm_cart, 2);
    }
  }

  MPI_Comm comm_cart;
  std::size_t global_resolution_x, global_resolution_y, global_resolution_z;
  std::size_t resolution_x, resolution_y, resolution_z;
  std::size_t offset_x, offset_y, offset_z;
};

template <class Stepper, class Analytical>
double run(Stepper &&stepper, mpi_setup const &setup, real_t tmax, real_t dt,
           Analytical &&exact) {
  const auto initial = analytical::to_domain(
      exact, setup.global_resolution_x, setup.global_resolution_y,
      setup.global_resolution_z, 0, setup.offset_x, setup.offset_y,
      setup.offset_z);
  solver_state state(setup.resolution_x, setup.resolution_y, setup.resolution_z,
                     initial.data(), initial.u(), initial.v(), initial.w());

  const halos_t halos{
      {{halo, halo, halo, halo + gt::uint_t(setup.resolution_x) - 1,
        halo + gt::uint_t(setup.resolution_x) + halo},
       {halo, halo, halo, halo + gt::uint_t(setup.resolution_y) - 1,
        halo + gt::uint_t(setup.resolution_y) + halo},
       {0, 0, 0, gt::uint_t(setup.resolution_z) - 1,
        gt::uint_t(setup.resolution_z)}}};

  const auto grid =
      gt::make_grid(halos[0], halos[1], axis_t{setup.resolution_z});
  const real_t dx = initial.dx;
  const real_t dy = initial.dy;
  const real_t dz = initial.dz;

  halo_exchange_f exchange{setup.comm_cart, state.sinfo};

  auto step = stepper(grid, exchange, dx, dy, dz);

  real_t t;
  for (t = 0; t < tmax; t += dt)
    step(state, dt);

  auto view = gt::make_host_view(state.data);

  const auto expected =
      analytical::to_domain(exact, setup.global_resolution_x,
                            setup.global_resolution_y,
                            setup.global_resolution_z, t, setup.offset_x,
                            setup.offset_y, setup.offset_z)
          .data();
  double error = 0.0;
#pragma omp parallel for reduction(+ : error)
  for (std::size_t i = halo; i < halo + setup.resolution_x; ++i) {
    for (std::size_t j = halo; j < halo + setup.resolution_y; ++j) {
      for (std::size_t k = 0; k < setup.resolution_z; ++k) {
        double diff = view(i, j, k) - expected(i, j, k);
        error += diff * diff;
      }
    }
  }

  MPI_Allreduce(&error, &error, 1, MPI_DOUBLE, MPI_SUM, setup.comm_cart);
  return error;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (argc != 6) {
    std::cerr << "usage: " << argv[0] << " NX NY NZ PX PY" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  mpi_setup setup(std::atoll(argv[1]), std::atoll(argv[2]), std::atoll(argv[3]),
                  std::atoi(argv[4]), std::atoi(argv[5]));

  storage_t::storage_info_t sinfo(setup.resolution_x + 2 * halo,
                                  setup.resolution_y + 2 * halo,
                                  setup.resolution_z);

  halo_exchange_f exchange{setup.comm_cart, sinfo};

  analytical::horizontal_diffusion exact{0.1};

  auto initial = analytical::to_domain(
      exact, setup.global_resolution_x, setup.global_resolution_y,
      setup.global_resolution_z, 0, setup.offset_x, setup.offset_y);

  storage_t storage(sinfo, initial.data(), "storage");

  exchange(storage);

  auto view = gt::make_host_view(storage);

  int rank, size;
  MPI_Comm_rank(setup.comm_cart, &rank);
  MPI_Comm_size(setup.comm_cart, &size);
  for (int r = 0; r < size; ++r) {
    if (r == rank) {
      std::cout << "rank " << rank << std::endl;
      for (std::size_t k = 0; k < setup.resolution_z; ++k) {
        for (std::size_t j = 0; j < setup.resolution_y + 2 * halo; ++j) {
          for (std::size_t i = 0; i < setup.resolution_x + 2 * halo; ++i)
            std::cout << view(i, j, k) << " ";
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
      std::cout.flush();
    }
    MPI_Barrier(setup.comm_cart);
  }

  MPI_Finalize();
}