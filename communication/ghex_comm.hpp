#pragma once

#include "communication.hpp"
#include "factorize.hpp"
#include <ghex/communication_object_2.hpp>
#include <ghex/structured/grid.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/threads/atomic/primitives.hpp>
#include <ghex/threads/std_thread/primitives.hpp>
#include <mpi.h>
#include <numeric>

#ifdef GTBENCH_USE_GHEX_UCP
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

#include <iostream>

namespace communication {

namespace ghex_comm {

using domain_id_t = int;
using dimension_t = std::integral_constant<int, 3>;
using coordinate_base_t = std::array<int, dimension_t::value>;
using coordinate_t = ::gridtools::ghex::coordinate<coordinate_base_t>;

struct local_domain {
public: // member types
  using domain_id_type = domain_id_t;
  using dimension = dimension_t;
  using coordinate_type = coordinate_t;

private: // members
  domain_id_type m_id;
  coordinate_type m_first;
  coordinate_type m_last;

public: // ctors
  template <typename Array>
  local_domain(domain_id_type id, const Array &first, const Array &last)
      : m_id{id} {
    std::copy(first.begin(), first.end(), m_first.begin());
    std::copy(last.begin(), last.end(), m_last.begin());
  }

  local_domain(const local_domain &) = default;
  local_domain(local_domain &&) = default;
  local_domain &operator=(const local_domain &) = default;
  local_domain &operator=(local_domain &&) = default;

public: // member functions
  domain_id_type domain_id() const { return m_id; }
  const coordinate_type &first() const { return m_first; }
  const coordinate_type &last() const { return m_last; }
};

using threading = gridtools::ghex::threads::std_thread::primitives;
using context_t = gridtools::ghex::tl::context<transport, threading>;
using communicator_t = context_t::communicator_type;
using grid_t =
    typename ::gridtools::ghex::structured::grid::template type<local_domain>;
using patterns_t =
    ::gridtools::ghex::pattern_container<communicator_t, grid_t, domain_id_t>;

struct halo_generator {
public: // member types
  using domain_type = local_domain;
  using dimension = typename domain_type::dimension;
  using coordinate_type = typename domain_type::coordinate_type;

  struct box {
    const coordinate_type &first() const { return m_first; }
    const coordinate_type &last() const { return m_last; }
    coordinate_type &first() { return m_first; }
    coordinate_type &last() { return m_last; }
    coordinate_type m_first;
    coordinate_type m_last;
  };

  struct box2 {
    const box &local() const { return m_local; }
    const box &global() const { return m_global; }
    box &local() { return m_local; }
    box &global() { return m_global; }
    box m_local;
    box m_global;
  };

private: // members
  coordinate_type m_first;
  coordinate_type m_last;

public: // ctors
  template <typename Array>
  halo_generator(const Array &g_first, const Array &g_last, int halo_size) {
    std::copy(g_first.begin(), g_first.end(), m_first.begin());
    std::copy(g_last.begin(), g_last.end(), m_last.begin());
  }

  halo_generator(const halo_generator &) = default;
  halo_generator(halo_generator &&) = default;
  halo_generator &operator=(const halo_generator &) = default;
  halo_generator &operator=(halo_generator &&) = default;

public: // member functions
  std::array<box2, 4> operator()(const domain_type &dom) const {
    // clang-format off
        coordinate_type my_first_local {                                0,                             -halo,                            0};
        coordinate_type my_first_global{                   dom.first()[0],               dom.first()[1]-halo,               dom.first()[2]};
        coordinate_type my_last_local  {     dom.last()[0]-dom.first()[0],                                -1, dom.last()[2]-dom.first()[2]};
        coordinate_type my_last_global {                    dom.last()[0],                  dom.first()[1]-1,                dom.last()[2]};

        coordinate_type mx_first_local {                            -halo,                                 0,                            0};
        coordinate_type mx_first_global{              dom.first()[0]-halo,                    dom.first()[1],               dom.first()[2]};
        coordinate_type mx_last_local  {                               -1,      dom.last()[1]-dom.first()[1], dom.last()[2]-dom.first()[2]};
        coordinate_type mx_last_global {                 dom.first()[0]-1,                     dom.last()[1],                dom.last()[2]};

        coordinate_type px_first_local {   dom.last()[0]-dom.first()[0]+1,                                 0,                            0};
        coordinate_type px_first_global{                  dom.last()[0]+1,                    dom.first()[1],               dom.first()[2]};
        coordinate_type px_last_local  {dom.last()[0]-dom.first()[0]+halo,      dom.last()[1]-dom.first()[1], dom.last()[2]-dom.first()[2]};
        coordinate_type px_last_global {               dom.last()[0]+halo,                     dom.last()[1],                dom.last()[2]};

        coordinate_type py_first_local {                                0,    dom.last()[1]-dom.first()[1]+1,                            0};
        coordinate_type py_first_global{                   dom.first()[0],                   dom.last()[1]+1,               dom.first()[2]};
        coordinate_type py_last_local  {     dom.last()[0]-dom.first()[0], dom.last()[1]-dom.first()[1]+halo, dom.last()[2]-dom.first()[2]};
        coordinate_type py_last_global {                    dom.last()[0],                dom.last()[1]+halo,                dom.last()[2]};

        my_first_global[1] = (((my_first_global[1]-m_first[1]) + (m_last[1]-m_first[1]+1)) % (m_last[1]-m_first[1]+1)) + m_first[1];
        my_last_global[1]  = ((( my_last_global[1]-m_first[1]) + (m_last[1]-m_first[1]+1)) % (m_last[1]-m_first[1]+1)) + m_first[1];

        mx_first_global[0] = (((mx_first_global[0]-m_first[0]) + (m_last[0]-m_first[0]+1)) % (m_last[0]-m_first[0]+1)) + m_first[0];
        mx_last_global[0]  = ((( mx_last_global[0]-m_first[0]) + (m_last[0]-m_first[0]+1)) % (m_last[0]-m_first[0]+1)) + m_first[0];

        px_first_global[0] = (((px_first_global[0]-m_first[0]) + (m_last[0]-m_first[0]+1)) % (m_last[0]-m_first[0]+1)) + m_first[0];
        px_last_global[0]  = ((( px_last_global[0]-m_first[0]) + (m_last[0]-m_first[0]+1)) % (m_last[0]-m_first[0]+1)) + m_first[0];

        py_first_global[1] = (((py_first_global[1]-m_first[1]) + (m_last[1]-m_first[1]+1)) % (m_last[1]-m_first[1]+1)) + m_first[1];
        py_last_global[1]  = ((( py_last_global[1]-m_first[1]) + (m_last[1]-m_first[1]+1)) % (m_last[1]-m_first[1]+1)) + m_first[1];

        return { 
            box2{ box{my_first_local, my_last_local}, box{my_first_global, my_last_global} },
            box2{ box{mx_first_local, mx_last_local}, box{mx_first_global, mx_last_global} },
            box2{ box{px_first_local, px_last_local}, box{px_first_global, px_last_global} },
            box2{ box{py_first_local, py_last_local}, box{py_first_global, py_last_global} }
        };
    // clang-format on
  }
};

struct world {

  struct moved_bit {
    bool m_moved = false;
    moved_bit() = default;
    moved_bit(const moved_bit &) = default;
    moved_bit(moved_bit &&other) noexcept
        : m_moved{std::exchange(other.m_moved, true)} {}
    moved_bit &operator=(const moved_bit &) = default;
    moved_bit &operator=(moved_bit &&other) noexcept {
      m_moved = std::exchange(other.m_moved, true);
      return *this;
    }
    operator bool() const { return m_moved; }
  };
  moved_bit m_moved;

  world(int &argc, char **&argv, bool mt = false) {
    if (mt) {
      int provided;
      int res = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
      if (res != MPI_SUCCESS)
        throw std::runtime_error("MPI init failed");
    } else {
      MPI_Init(&argc, &argv);
    }

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size > 1 && rank != 0)
      std::cout.setstate(std::ios_base::failbit);
  }

  world(world const &) = delete;
  world &operator=(world const &) = delete;
  world(world &&other) = default;
  world &operator=(world &&other) = default;

  ~world() {
    if (!m_moved)
      MPI_Finalize();
  }
};

class grid {
public: // member types
  using domain_id_type = typename local_domain::domain_id_type;
  using coordinate_type = typename local_domain::coordinate_type;
  using patterns_type = patterns_t;
  using patterns_ptr_t = std::unique_ptr<patterns_type>;
  using comm_obj_type =
      ::gridtools::ghex::communication_object<communicator_t, grid_t,
                                              domain_id_t>;
  using comm_obj_ptr_t = std::unique_ptr<comm_obj_type>;
  using domain_vec = std::vector<local_domain>;
  using context_ptr_t = std::unique_ptr<context_t>;
  using thread_token = context_t::thread_token;

  struct sub_grid {
    int m_rank;
    int m_size;
    domain_id_type m_domain_id;
    context_t *m_context;
    patterns_type *m_patterns;
    mutable thread_token m_token;
    comm_obj_ptr_t m_comm_obj;
    vec<std::size_t, 2> global_resolution;
    vec<std::size_t, 2> offset;
    vec<std::size_t, 3> resolution;
  };

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
  grid(vec<std::size_t, 3> const &global_resolution, int num_sub_domains = 1)
      : m_hg{std::array<int, 3>{0, 0, 0},
             std::array<int, 3>{(int)global_resolution.x - 1,
                                (int)global_resolution.y - 1,
                                (int)global_resolution.z - 1},
             halo},
        m_global_resolution{global_resolution.x, global_resolution.y},
        m_tokens(num_sub_domains) {
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    std::array<int, 2> m_coords;

    // divide the domain into m_size sub-domains
    const auto div_ranks =
        divide_domain(m_size, std::array<std::size_t, 2>{global_resolution.x,
                                                         global_resolution.y});
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
                                   (std::size_t)(m_last[1] - m_first[1] + 1)});
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

    m_context =
        gridtools::ghex::tl::context_factory<transport, threading>::create(
            num_sub_domains, MPI_COMM_WORLD);
    m_patterns = std::make_unique<patterns_type>(
        ::gridtools::ghex::make_pattern<::gridtools::ghex::structured::grid>(
            *m_context, m_hg, m_domains));
  }

  sub_grid operator[](unsigned int i) {
    const auto &dom = m_domains[i];
    if (!m_tokens[i])
      m_tokens[i] = std::make_unique<thread_token>(m_context->get_token());
    auto comm = m_context->get_communicator(*m_tokens[i]);
    return {
        m_rank,
        m_size,
        dom.domain_id(),
        m_context.get(),
        m_patterns.get(),
        *m_tokens[i],
        comm_obj_ptr_t{new comm_obj_type{
            ::gridtools::ghex::make_communication_object<patterns_type>(comm)}},
        m_global_resolution,
        {(std::size_t)dom.first()[0], (std::size_t)dom.first()[1]},
        {(std::size_t)(dom.last()[0] - dom.first()[0] + 1),
         (std::size_t)(dom.last()[1] - dom.first()[1] + 1),
         (std::size_t)(dom.last()[2] - dom.first()[2] + 1)}};
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

inline grid comm_grid(const world &,
                      vec<std::size_t, 3> const &global_resolution,
                      int num_threads) {
  return {global_resolution, num_threads};
}

inline grid::sub_grid comm_sub_grid(grid &grid, int id = 0) { return grid[id]; }

std::function<void(storage_t &)>
comm_halo_exchanger(grid::sub_grid &grid,
                    storage_t::storage_info_t const &sinfo);

double comm_global_max(grid::sub_grid const &grid, double t);

} // namespace ghex_comm

} // namespace communication
