#pragma once

#include <mpi.h>

#include "communication.hpp"
#include <ghex/communication_object_2.hpp>
#include <ghex/structured/grid.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/transport_layer/mpi/communicator.hpp>

#include <iostream>

namespace communication {

namespace ghex_comm {

struct local_domain {
public: // member types
  using domain_id_type = int;
  using dimension = std::integral_constant<int, 3>;
  using coordinate_base_type = std::array<int, dimension::value>;
  using coordinate_type = ::gridtools::ghex::coordinate<coordinate_base_type>;

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

  bool m_moved = false;

  world(int &argc, char **&argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size > 1 && rank != 0)
      std::cout.setstate(std::ios_base::failbit);
  }

  world(world const &) = delete;

  world &operator=(world const &) = delete;

  world(world &&other) noexcept : m_moved{other.m_moved} {
    other.m_moved = true;
  }

  world &operator=(world &&other) noexcept {
    // if (!m_moved)
    m_moved = other.m_moved;
    other.m_moved = true;
    return *this;
  }

  ~world() {
    if (!m_moved)
      MPI_Finalize();
  }
};

struct grid {
public: // member types
  using transport_type = ::gridtools::ghex::tl::mpi_tag;
  using domain_id_type = typename local_domain::domain_id_type;
  using coordinate_type = typename local_domain::coordinate_type;
  using grid_type =
      typename ::gridtools::ghex::structured::grid::template type<local_domain>;
  using patterns_type =
      ::gridtools::ghex::pattern_container<transport_type, grid_type,
                                           domain_id_type>;
  using patterns_ptr_t = std::unique_ptr<patterns_type>;
  using comm_obj_type =
      typename std::remove_reference<typename std::remove_cv<decltype(
          ::gridtools::ghex::make_communication_object<patterns_type>())>::
                                         type>::type;

private: // members
  int m_rank;
  int m_size;
  std::array<int, 2> m_dims;
  MPI_Comm m_comm_cart;
  std::array<int, 2> m_coords;
  coordinate_type m_first;
  coordinate_type m_last;
  local_domain m_dom;
  halo_generator m_hg;
  patterns_ptr_t m_patterns;
  bool m_moved = false;

public: // members
  vec<std::size_t, 2> global_resolution;
  vec<std::size_t, 2> offset;
  vec<std::size_t, 3> resolution;

public: // ctors
  grid(vec<std::size_t, 3> const &global_resolution)
      : m_rank{[]() {
          int r;
          MPI_Comm_rank(MPI_COMM_WORLD, &r);
          return r;
        }()},
        m_size{[]() {
          int s;
          MPI_Comm_size(MPI_COMM_WORLD, &s);
          return s;
        }()},
        m_dims{[this]() {
          std::array<int, 2> dims = {0, 0};
          MPI_Dims_create(this->m_size, 2, dims.data());
          return dims;
        }()},
        m_comm_cart{[this]() {
          MPI_Comm comm_;
          std::array<int, 2> periods = {1, 1};
          MPI_Cart_create(MPI_COMM_WORLD, 2, m_dims.data(), periods.data(), 1,
                          &comm_);
          return comm_;
        }()},
        m_coords{[this]() {
          std::array<int, 2> coords;
          MPI_Cart_coords(m_comm_cart, m_rank, 2, coords.data());
          return coords;
        }()},
        m_first{(int)global_resolution.x * m_coords[0] / m_dims[0],
                (int)global_resolution.y * m_coords[1] / m_dims[1], 0},
        m_last{(int)global_resolution.x * (m_coords[0] + 1) / m_dims[0] - 1,
               (int)global_resolution.y * (m_coords[1] + 1) / m_dims[1] - 1,
               (int)global_resolution.z - 1},
        m_dom{m_rank, m_first, m_last}, m_hg{std::array<int, 3>{0, 0, 0},
                                             std::array<int, 3>{
                                                 (int)global_resolution.x - 1,
                                                 (int)global_resolution.y - 1,
                                                 (int)global_resolution.z - 1},
                                             halo},
        m_patterns{new patterns_type{::gridtools::ghex::make_pattern<
            ::gridtools::ghex::structured::grid>(
            m_comm_cart, m_hg, std::vector<local_domain>{m_dom})}},
        global_resolution{global_resolution.x, global_resolution.y},
        offset{(std::size_t)m_first[0], (std::size_t)m_first[1]},
        resolution{(std::size_t)(m_last[0] - m_first[0] + 1),
                   (std::size_t)(m_last[1] - m_first[1] + 1),
                   global_resolution.z} {
    if (resolution.x < halo || resolution.y < halo)
      throw std::runtime_error("local resolution is smaller than halo size!");
  }

  grid(grid const &) = delete;

  grid(grid &&other) noexcept
      : m_rank{other.m_rank}, m_size{other.m_size}, m_dims{other.m_dims},
        m_comm_cart{other.m_comm_cart}, m_coords{other.m_coords},
        m_first{other.m_first}, m_last{other.m_last}, m_dom{other.m_dom},
        m_hg{other.m_hg}, m_patterns{std::move(other.m_patterns)},
        global_resolution{other.global_resolution}, offset{other.offset},
        resolution{other.resolution} {
    other.m_moved = true;
  }

  ~grid() {
    if (!m_moved)
      MPI_Comm_free(&m_comm_cart);
  }

public: // member functions
  int rank() const noexcept { return m_rank; }
  int size() const noexcept { return m_size; }

  const patterns_type &patterns() const { return *m_patterns; }
  patterns_type &patterns() { return *m_patterns; }

  domain_id_type domain_id() const { return m_dom.domain_id(); }

  MPI_Comm mpi_comm() const { return m_comm_cart; }
};

inline grid comm_grid(const world &,
                      vec<std::size_t, 3> const &global_resolution) {
  return {global_resolution};
}

std::function<void(storage_t &)>
comm_halo_exchanger(grid const &grid, storage_t::storage_info_t const &sinfo);

double comm_global_max(grid const &grid, double t);

} // namespace ghex_comm

} // namespace communication
