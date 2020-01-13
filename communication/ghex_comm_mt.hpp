#pragma once

#include "factorize.hpp"
#include "ghex_comm.hpp"
#include <numeric>

#include <ghex/threads/atomic/primitives.hpp>
namespace communication {

namespace ghex_comm {

using threading_mt = gridtools::ghex::threads::atomic::primitives;
using context_mt_t = gridtools::ghex::tl::context<transport, threading_mt>;
using communicator_mt_t = context_mt_t::communicator_type;
using patterns_mt_t = ::gridtools::ghex::pattern_container<communicator_mt_t,
                                                           grid_t, domain_id_t>;

class grid_mt {
public: // member types
  using domain_id_type = typename local_domain::domain_id_type;
  using coordinate_type = typename local_domain::coordinate_type;
  using patterns_type = patterns_mt_t;
  using patterns_ptr_t = std::unique_ptr<patterns_type>;
  using comm_obj_type =
      ::gridtools::ghex::communication_object<communicator_mt_t, grid_t,
                                              domain_id_t>;
  using comm_obj_ptr_t = std::unique_ptr<comm_obj_type>;
  using domain_vec = std::vector<local_domain>;
  using context_ptr_t = std::unique_ptr<context_mt_t>;
  using thread_token = context_mt_t::thread_token;

  struct sub_grid {
    int m_rank;
    int m_size;
    domain_id_type m_domain_id;
    context_mt_t *m_context;
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
  // moved_bit           m_moved;

public:
  grid_mt(vec<std::size_t, 3> const &global_resolution, int num_sub_domains = 1)
      : m_hg{std::array<int, 3>{0, 0, 0},
             std::array<int, 3>{(int)global_resolution.x - 1,
                                (int)global_resolution.y - 1,
                                (int)global_resolution.z - 1},
             halo},
        m_global_resolution{global_resolution.x, global_resolution.y} {
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
        gridtools::ghex::tl::context_factory<transport, threading_mt>::create(
            num_sub_domains, MPI_COMM_WORLD);
    m_patterns = std::make_unique<patterns_type>(
        ::gridtools::ghex::make_pattern<::gridtools::ghex::structured::grid>(
            *m_context, m_hg, m_domains));
  }

  sub_grid operator[](unsigned int i) {
    const auto &dom = m_domains[i];

    auto t = m_context->get_token();
    auto comm = m_context->get_communicator(t);
    return {
        m_rank,
        m_size,
        dom.domain_id(),
        m_context.get(),
        m_patterns.get(),
        t,
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

std::function<void(storage_t &)>
comm_halo_exchanger(grid_mt::sub_grid &grid,
                    storage_t::storage_info_t const &sinfo);

double comm_global_max(grid_mt::sub_grid const &grid, double t);

} // namespace ghex_comm
} // namespace communication
