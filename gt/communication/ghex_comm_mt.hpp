#pragma once

#include "./ghex_comm.hpp"
#include "./factorize.hpp"
#include <ghex/glue/gridtools/field.hpp>

namespace communication {

namespace ghex_comm {


class grid_mt
{
public: // member types
  using domain_id_type    = local_domain::domain_id_type;
  using transport_type    = ::gridtools::ghex::tl::mpi_tag;
  using grid_type         = ::gridtools::ghex::structured::grid::template type<local_domain>;
  using patterns_type     = ::gridtools::ghex::pattern_container<transport_type, grid_type, domain_id_type>;
  using patterns_ptr_t    = std::unique_ptr<patterns_type>;

protected: // member types
    using coordinate_type = local_domain::coordinate_type;
    using domain_vec      = std::vector<local_domain>;
    using comm_obj_type   = typename std::remove_reference<typename std::remove_cv<decltype(
                               ::gridtools::ghex::make_communication_object<patterns_type>())>::type>::type;
    using comm_obj_ptr_t  = std::unique_ptr<comm_obj_type>;

public:
    struct sub_grid
    {
        domain_id_type      domain_id;
        MPI_Comm            mpi_comm;
        comm_obj_ptr_t      co;
        patterns_type*      patterns;
        vec<std::size_t, 2> global_resolution;
        vec<std::size_t, 2> offset;
        vec<std::size_t, 3> resolution;
    };

protected: // members
    int                 m_size;
    std::array<int, 2>  m_dims;
    MPI_Comm            m_comm_cart;
    int                 m_rank;
    std::array<int, 2>  m_coords;
    coordinate_type     m_first;
    coordinate_type     m_last;
    halo_generator      m_hg;
    vec<std::size_t, 2> m_global_resolution;
    domain_vec          m_domains;
    patterns_ptr_t      m_patterns;
    moved_bit           m_moved;

public: // ctors
    grid_mt(vec<std::size_t, 3> const &global_resolution, int num_sub_domains = 1)
    : m_size{[this]()
        {
            int s;
            MPI_Comm_size(MPI_COMM_WORLD, &s);
            return s;
        }()}
    , m_dims{[this]()
        {
            std::array<int, 2> dims = {0, 0};
            MPI_Dims_create(this->m_size, 2, dims.data());
            return dims;
        }()}
    , m_comm_cart{[this]()
        {
            MPI_Comm comm_;
            std::array<int, 2> periods = {1, 1};
            MPI_Cart_create(MPI_COMM_WORLD, 2, m_dims.data(), periods.data(), 1, &comm_);
            return comm_;
        }()}
    , m_rank{[this]()
        {
            int r;
            MPI_Comm_rank(m_comm_cart, &r);
            return r;
        }()}
    , m_coords{[this]()
        {
            std::array<int, 2> coords;
            MPI_Cart_coords(m_comm_cart, m_rank, 2, coords.data());
            return coords;
        }()}
    , m_first{(int)global_resolution.x * m_coords[0] / m_dims[0],
              (int)global_resolution.y * m_coords[1] / m_dims[1], 
              0}
    , m_last{ (int)global_resolution.x * (m_coords[0] + 1) / m_dims[0] - 1,
              (int)global_resolution.y * (m_coords[1] + 1) / m_dims[1] - 1,
              (int)global_resolution.z - 1}
    , m_hg{std::array<int, 3>{0, 0, 0},
           std::array<int, 3>{(int)global_resolution.x - 1, (int)global_resolution.y - 1, (int)global_resolution.z - 1}, 
           halo}
    , m_global_resolution{global_resolution.x, global_resolution.y}
    {
        m_domains.reserve(num_sub_domains);

        auto div = divide_domain(num_sub_domains, std::array<int,2>{m_last[0]-m_first[0]+1,m_last[1]-m_first[1]+1});

        int i = 0;
        int y = m_first[1];
        for (const auto& dy : div[1])
        {
            int x = m_first[0];
            for (const auto& dx : div[0])
            {
                //std::cout << "[(" << x << ", " << y << "), (" << x+dx-1 << ", " << y+dy-1 << ")]" << std::endl;
                m_domains.push_back( local_domain{ 
                    m_rank*num_sub_domains+i,
                    coordinate_type{x,y,m_first[2]},
                    coordinate_type{x+dx-1,y+dy-1,m_last[2]} });
                x+=dx;
                ++i;
            }
            y+=dy; 
        }
        
        m_patterns.reset( new patterns_type{
            ::gridtools::ghex::make_pattern<::gridtools::ghex::structured::grid>(m_comm_cart, m_hg, m_domains)});
    }

    grid_mt(const grid_mt&) = delete;
    grid_mt(grid_mt&&) = default;
    ~grid_mt() { if (!m_moved) MPI_Comm_free(&m_comm_cart); }

public: // member functions

    sub_grid operator[](unsigned int i) const
    {
        const auto& dom = m_domains[i];

        return { dom.domain_id(),
                 m_comm_cart,
                 comm_obj_ptr_t{new comm_obj_type{::gridtools::ghex::make_communication_object<patterns_type>()}},
                 m_patterns.get(),
                 m_global_resolution, 
                 {(std::size_t)dom.first()[0],
                  (std::size_t)dom.first()[1]},
                 {(std::size_t)(dom.last()[0]-dom.first()[0]+1),
                  (std::size_t)(dom.last()[1]-dom.first()[1]+1),
                  (std::size_t)(dom.last()[2]-dom.first()[2]+1)}};

    }

protected: // implementation
};

std::function<void(storage_t &)>
comm_halo_exchanger(grid_mt::sub_grid &g, storage_t::storage_info_t const &sinfo)
{
    auto co_ptr          = g.co.get();
    auto patterns_ptr    = g.patterns;
    const auto domain_id = g.domain_id;
    
    return [co_ptr, patterns_ptr, domain_id](const storage_t &storage) mutable
    {
        auto &co = *co_ptr;
        auto &patterns = *patterns_ptr;
        auto field = ::gridtools::ghex::wrap_gt_field(domain_id, storage);

        #ifdef __CUDACC__
        cudaDeviceSynchronize();
        #endif

        co.exchange(patterns(field)).wait();
    };
}

double comm_global_max(grid_mt::sub_grid const &g, double t) {
  double max;
  MPI_Allreduce(&t, &max, 1, MPI_DOUBLE, MPI_MAX, g.mpi_comm);
  return max;
}

} // namespace ghex_comm
} // namespace communication

