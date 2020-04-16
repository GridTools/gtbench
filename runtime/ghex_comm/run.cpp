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
#include "./factorize.hpp"

#include <mpi.h>
#include <numeric>
#include <regex>

#include <ghex/communication_object_2.hpp>
#include <ghex/glue/gridtools/field.hpp>
#include <ghex/structured/grid.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/threads/atomic/primitives.hpp>
#include <ghex/threads/std_thread/primitives.hpp>

#ifdef GTBENCH_USE_GHEX_UCP
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gt::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gt::ghex::tl::mpi_tag;
#endif

namespace runtime {

namespace ghex_comm_impl {

runtime::runtime(int num_threads, std::array<int, 2> cart_dims,
                 std::array<int, 2> thread_cart_dims,
                 std::vector<int> const &device_mapping)
    : m_scope(
          [=] {
            if (num_threads > 1) {
              int provided;
              MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            } else {
              MPI_Init(nullptr, nullptr);
            }
          },
          MPI_Finalize),
      m_num_threads(num_threads), m_cart_dims(cart_dims),
      m_thread_cart_dims(thread_cart_dims), m_device_mapping(num_threads, 0) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size > 1 && rank != 0)
    std::cout.setstate(std::ios_base::failbit);

  MPI_Dims_create(size, 2, m_cart_dims.data());
  if (m_cart_dims[0] * m_cart_dims[1] != size) {
    throw std::runtime_error(
        "the product of cart dims must be equal to the number of MPI ranks.");
  }
  if (m_thread_cart_dims[0] * m_thread_cart_dims[1] != num_threads) {
    if ((m_thread_cart_dims[0] != 0 && m_thread_cart_dims[1] != 0) ||
        m_thread_cart_dims[0] > num_threads ||
        m_thread_cart_dims[1] > num_threads)
      throw std::runtime_error(
          "the product of thread cart dims must be equal to the number of "
          "threads per rank.");
    if (m_thread_cart_dims[0] == 0 && m_thread_cart_dims[1] == 0)
      m_thread_cart_dims =
          partition_factors(factorize(num_threads), std::array<int, 2>{1, 1});
    else if (m_thread_cart_dims[0] == 0)
      m_thread_cart_dims[0] = num_threads / m_thread_cart_dims[1];
    else
      m_thread_cart_dims[1] = num_threads / m_thread_cart_dims[0];
    if (m_thread_cart_dims[0] * m_thread_cart_dims[1] != num_threads)
      throw std::runtime_error(
          "the product of thread cart dims must be equal to the number of "
          "threads per rank.");
  }

#ifdef __CUDACC__
  MPI_Comm shmem_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shmem_comm);
  int shmem_size, shmem_rank;
  MPI_Comm_size(shmem_comm, &shmem_size);
  MPI_Comm_rank(shmem_comm, &shmem_rank);
  MPI_Comm_free(&shmem_comm);
  if (!device_mapping.empty()) {
    if (device_mapping.size() != shmem_size * num_threads)
      throw std::runtime_error("device mapping has wrong size");
    m_device_mapping = device_mapping;
  } else {
    m_device_mapping.resize(shmem_size * m_num_threads);
    std::iota(m_device_mapping.begin(), m_device_mapping.end(), 0);
  }
  m_device_mapping = std::vector<int>(
      m_device_mapping.begin() + shmem_rank * num_threads,
      m_device_mapping.begin() + (shmem_rank + 1) * num_threads);
#endif
}

using domain_id_t = int;
using dimension_t = std::integral_constant<int, 3>;
using coordinate_t = gt::ghex::coordinate<std::array<int, 3>>;

struct local_domain {
  using domain_id_type = domain_id_t;
  using dimension = dimension_t;
  using coordinate_type = coordinate_t;

  domain_id_t m_domain_id;
  coordinate_t m_first;
  coordinate_t m_last;

  domain_id_t domain_id() const { return m_domain_id; }
  coordinate_t const &first() const { return m_first; }
  coordinate_t const &last() const { return m_last; }
};

using threading = gt::ghex::threads::std_thread::primitives;
using context_t = gt::ghex::tl::context<transport, threading>;
using communicator_t = context_t::communicator_type;
using grid_t = gt::ghex::structured::grid::type<local_domain>;
using patterns_t =
    gt::ghex::pattern_container<communicator_t, grid_t, domain_id_t>;

struct halo_generator {
  using domain_type = local_domain;
  using dimension = dimension_t;
  using coordinate_type = coordinate_t;

  struct box {
    const coordinate_t &first() const { return m_first; }
    const coordinate_t &last() const { return m_last; }
    coordinate_t m_first;
    coordinate_t m_last;
  };

  struct box2 {
    const box &local() const { return m_local; }
    const box &global() const { return m_global; }
    box m_local;
    box m_global;
  };

  coordinate_t m_first;
  coordinate_t m_last;

  std::array<box2, 4> operator()(const domain_type &dom) const {
    coordinate_t res = {dom.last()[0] - dom.first()[0],
                        dom.last()[1] - dom.first()[1],
                        dom.last()[2] - dom.first()[2]};

    box mx_local = {{-halo, 0, 0}, {-1, res[1], res[2]}};
    box my_local = {{0, -halo, 0}, {res[0], -1, res[2]}};
    box px_local = {{res[0] + 1, 0, 0}, {res[0] + halo, res[1], res[2]}};
    box py_local = {{0, res[1] + 1, 0}, {res[0], res[1] + halo, res[2]}};

    auto local_to_global = [&](coordinate_t const &local, std::size_t axis) {
      coordinate_t global = {local[0] + dom.first()[0],
                             local[1] + dom.first()[1],
                             local[2] + dom.first()[2]};
      auto res = m_last[axis] - m_first[axis] + 1;
      global[axis] = (global[axis] - m_first[axis] + res) % res + m_first[axis];
      return global;
    };

    auto make_local_global_pair = [&](box const &local, std::size_t axis) {
      return box2{local,
                  {local_to_global(local.first(), axis),
                   local_to_global(local.last(), axis)}};
    };

    return {make_local_global_pair(my_local, 1),
            make_local_global_pair(mx_local, 0),
            make_local_global_pair(px_local, 0),
            make_local_global_pair(py_local, 1)};
  }
};

class grid::impl {
public: // member types
  using domain_id_type = local_domain::domain_id_type;
  using coordinate_type = local_domain::coordinate_type;
  using patterns_type = patterns_t;
  using patterns_ptr_t = std::unique_ptr<patterns_type>;
  using comm_obj_type =
      gt::ghex::communication_object<communicator_t, grid_t, domain_id_t>;
  using comm_obj_ptr_t = std::unique_ptr<comm_obj_type>;
  using domain_vec = std::vector<local_domain>;
  using context_ptr_t = std::unique_ptr<context_t>;
  using thread_token = context_t::thread_token;

private: // members
  halo_generator m_hg;
  vec<std::size_t, 2> m_global_resolution;
  int m_size;
  int m_rank;
  coordinate_type m_first;
  coordinate_type m_last;
  domain_vec m_domains;
  context_ptr_t m_context;
  patterns_ptr_t m_patterns;
  std::vector<std::unique_ptr<thread_token>> m_tokens;

public:
  impl(vec<std::size_t, 3> const &global_resolution, int num_sub_domains,
       std::array<int, 2> cart_dims, std::array<int, 2> thread_cart_dims)
      : m_hg{std::array<int, 3>{0, 0, 0},
             std::array<int, 3>{(int)global_resolution.x - 1,
                                (int)global_resolution.y - 1,
                                (int)global_resolution.z - 1}},
        m_global_resolution{global_resolution.x, global_resolution.y},
        m_tokens(num_sub_domains) {
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    std::array<int, 2> m_coords;

    // divide the domain into m_size sub-domains
    const auto div_ranks = divide_domain(
        m_size,
        std::array<std::size_t, 2>{global_resolution.x, global_resolution.y},
        cart_dims);
    // compute the offsets
    std::array<std::vector<std::size_t>, 2> offsets_ranks = {
        compute_offsets(div_ranks[0], 0), compute_offsets(div_ranks[1], 0)};
    // compute the rank coordinates of my sub-domain
    const auto n_x = div_ranks[0].size();
    m_coords[1] = (m_rank / n_x);
    m_coords[0] = m_rank - m_coords[1] * n_x;
    // compute the global coordinates of my sub-domain
    m_first[0] = offsets_ranks[0][m_coords[0]];
    m_first[1] = offsets_ranks[1][m_coords[1]];
    m_last[0] = offsets_ranks[0][m_coords[0] + 1] - 1;
    m_last[1] = offsets_ranks[1][m_coords[1] + 1] - 1;
    // divide my sub-domain further into num_sub_domanis parts
    const auto div_threads = divide_domain(
        num_sub_domains,
        std::array<std::size_t, 2>{(std::size_t)(m_last[0] - m_first[0] + 1),
                                   (std::size_t)(m_last[1] - m_first[1] + 1)},
        thread_cart_dims);
    // compute the offsets
    std::array<std::vector<std::size_t>, 2> offsets_threads = {
        compute_offsets(div_threads[0], m_first[0]),
        compute_offsets(div_threads[1], m_first[1])};

    // make domains
    int i = 0;
    for (std::size_t y = 0; y < div_threads[1].size(); ++y) {
      for (std::size_t x = 0; x < div_threads[0].size(); ++x, ++i) {
        m_domains.push_back(
            local_domain{m_rank * num_sub_domains + i,
                         coordinate_type{(int)(offsets_threads[0][x]),
                                         (int)(offsets_threads[1][y]), 0},
                         coordinate_type{(int)(offsets_threads[0][x + 1] - 1),
                                         (int)(offsets_threads[1][y + 1] - 1),
                                         (int)global_resolution.z - 1}});
      }
    }

    m_context = gt::ghex::tl::context_factory<transport, threading>::create(
        num_sub_domains, MPI_COMM_WORLD);
    m_patterns = std::make_unique<patterns_type>(
        gt::ghex::make_pattern<gt::ghex::structured::grid>(*m_context, m_hg,
                                                           m_domains));
  }

  impl(impl const &) = delete;
  impl &operator=(impl const &) = delete;

  sub_grid operator[](unsigned int i) {
    const auto &dom = m_domains[i];
    if (!m_tokens[i])
      m_tokens[i] = std::make_unique<thread_token>(m_context->get_token());
    auto comm = m_context->get_communicator(*m_tokens[i]);
    comm.barrier();

    vec<std::size_t, 3> local_resolution = {
        (std::size_t)(dom.last()[0] - dom.first()[0] + 1),
        (std::size_t)(dom.last()[1] - dom.first()[1] + 1),
        (std::size_t)(dom.last()[2] - dom.first()[2] + 1)};
    vec<std::size_t, 3> local_offset = {(std::size_t)dom.first()[0],
                                        (std::size_t)dom.first()[1],
                                        (std::size_t)dom.first()[2]};

    auto comm_obj = std::make_shared<comm_obj_type>(
        gt::ghex::make_communication_object<patterns_type>(comm));

    auto halo_exchange = [comm_obj = std::move(comm_obj),
                          domain_id = dom.domain_id(),
                          &patterns = *m_patterns](storage_t &storage) mutable {
      auto field = gt::ghex::wrap_gt_field(domain_id, storage);

#ifdef __CUDACC__
      cudaStreamSynchronize(0);
#endif
      comm_obj->exchange(patterns(field)).wait();
    };

    return {local_resolution, local_offset, std::move(halo_exchange)};
  }

  result collect_results(result r) const {
    result reduced;
    MPI_Allreduce(&r, &reduced, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return reduced;
  }

  std::vector<std::size_t> compute_offsets(const std::vector<std::size_t> &dx,
                                           std::size_t x_0) const {
    std::vector<std::size_t> offsets(dx.size() + 1, 0);
    std::partial_sum(dx.begin(), dx.end(), offsets.begin() + 1);
    for (auto &o : offsets)
      o += x_0;
    return offsets;
  }
};

grid::grid(vec<std::size_t, 3> const &global_resolution, int num_sub_domains,
           std::array<int, 2> cart_dims, std::array<int, 2> thread_cart_dims)
    : m_impl(std::make_unique<impl>(global_resolution, num_sub_domains,
                                    cart_dims, thread_cart_dims)) {}

grid::~grid() {}

sub_grid grid::operator[](unsigned id) { return (*m_impl)[id]; }

result grid::collect_results(result const &r) const {
  return m_impl->collect_results(r);
}

void runtime_register_options(ghex_comm, options &options) {
  options("cart-dims", "dimensons of cartesian communicator", "PX PY", 2);
  options("sub-domains",
          "number of sub-domains (each sub-domain computation runs in its own "
          "thread)",
          "S", {1});
  options("thread-cart-dims",
          "dimensons of cartesian decomposition "
          "among sub-domains",
          "TX TY", 2);
#ifdef __CUDACC__
  options("device-mapping",
          "node device mapping: device id per sub-domain in the format "
          "I_0:I_1:...:I_(N-1) "
          "where I_i are cuda device ids "
          "and N = #ranks-per-node x S",
          "M");
#endif
}

runtime runtime_init(ghex_comm, options_values const &options) {
  std::array<int, 2> cart_dims = {0, 0};
  if (options.has("cart-dims"))
    cart_dims = options.get<std::array<int, 2>>("cart-dims");
  std::array<int, 2> thread_cart_dims = {0, 0};
  if (options.has("thread-cart-dims"))
    thread_cart_dims = options.get<std::array<int, 2>>("thread-cart-dims");
#ifdef __CUDACC__
  std::vector<int> device_mapping;
  if (options.has("device-mapping")) {
    const std::regex delimiter(":");
    const auto input = options.get<std::string>("device-mapping");
    std::transform(
        std::sregex_token_iterator(input.begin(), input.end(), delimiter, -1),
        std::sregex_token_iterator(), std::back_inserter(device_mapping),
        [](const std::string &token) {
          const int n = std::stoi(token);
          if (n < 0)
            throw std::runtime_error("negative device ids are not allowed");
          return n;
        });
  }
  return runtime(options.get<int>("sub-domains"), cart_dims, thread_cart_dims,
                 device_mapping);
#else
  return runtime(options.get<int>("sub-domains"), cart_dims, thread_cart_dims);
#endif
}

} // namespace ghex_comm_impl

} // namespace runtime
