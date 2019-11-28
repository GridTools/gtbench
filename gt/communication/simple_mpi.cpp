#include "./simple_mpi.hpp"

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

grid::grid(vec<std::size_t, 3> const &global_resolution)
    : global_resolution{global_resolution.x, global_resolution.y} 
{
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::array<int, 2> dims = {0, 0}, periods = {1, 1};
  MPI_Dims_create(size, 2, dims.data());
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1,
                  &comm_cart);
  std::array<int, 2> coords;
  MPI_Cart_coords(comm_cart, rank, 2, coords.data());

  offset.x = global_resolution.x * coords[0] / dims[0];
  offset.y = global_resolution.y * coords[1] / dims[1];

  vec<std::size_t, 2> next_offset = {
      global_resolution.x * (coords[0] + 1) / dims[0],
      global_resolution.y * (coords[1] + 1) / dims[1]};

  resolution.x = next_offset.x - offset.x;
  resolution.y = next_offset.y - offset.y;
  resolution.z = global_resolution.z;

  if (resolution.x < halo || resolution.y < halo)
    throw std::runtime_error("local resolution is smaller than halo size!");
}

grid::~grid() { MPI_Comm_free(&comm_cart); }

template <class T> struct halo_info { T lower, upper; };

struct halo_exchange_f {
  halo_exchange_f(MPI_Comm comm_cart, storage_t::storage_info_t const &sinfo)
      : comm_cart(comm_cart), send_offsets{{0, 0}, {0, 0}}, recv_offsets{{0, 0}, {0, 0}} 
  {
    auto strides = sinfo.strides();
    auto sizes = sinfo.total_lengths();

    // sized of halos to exchange along x- and y-axes
    vec<decltype(sizes), 2> halo_sizes;
    halo_sizes.x = {halo, sizes[1] - 2 * halo, sizes[2]};
    halo_sizes.y = {sizes[0] - 2 * halo, halo, sizes[2]};

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
    for (int i = 0; i < 3; ++i)
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

    // building of halo data type for exchanges along x-axis
    halo_dtypes.x =
        std::shared_ptr<MPI_Datatype>(new MPI_Datatype, mpi_dtype_deleter);
    MPI_Type_hvector(halo_sizes.x[1], halo_sizes.x[0],
                     strides[1] * sizeof(real_t), mpi_real_dtype, &plane);
    MPI_Type_hvector(halo_sizes.x[2], 1, strides[2] * sizeof(real_t), plane,
                     halo_dtypes.x.get());
    MPI_Type_commit(halo_dtypes.x.get());
    MPI_Type_free(&plane);

    // building of halo data type for exchanges along y-axis
    halo_dtypes.y =
        std::shared_ptr<MPI_Datatype>(new MPI_Datatype, mpi_dtype_deleter);
    MPI_Type_hvector(halo_sizes.y[1], halo_sizes.y[0],
                     strides[1] * sizeof(real_t), mpi_real_dtype, &plane);
    MPI_Type_hvector(halo_sizes.y[2], 1, strides[2] * sizeof(real_t), plane,
                     halo_dtypes.y.get());
    MPI_Type_commit(halo_dtypes.y.get());
    MPI_Type_free(&plane);
  }

  void operator()(storage_t &storage) const {
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
  }

  // cartesian communicator
  MPI_Comm comm_cart;
  // MPI dtypes for halos along x- and y-axes (shared ptr to make the functor
  // copyable and thus usable inside std::function<>)
  vec<std::shared_ptr<MPI_Datatype>, 2> halo_dtypes;
  // send and recv offsets for upper and lower halos
  vec<halo_info<std::size_t>, 2> send_offsets, recv_offsets;
};

std::function<void(storage_t &)>
comm_halo_exchanger(grid const &comm_grid,
                    storage_t::storage_info_t const &sinfo) {
  return halo_exchange_f(comm_grid.comm_cart, sinfo);
}

double comm_global_max(grid const &grid, double t) {
  double max;
  MPI_Allreduce(&t, &max, 1, MPI_DOUBLE, MPI_SUM, grid.comm_cart);
  return max;
}

} // namespace simple_mpi

} // namespace communication

