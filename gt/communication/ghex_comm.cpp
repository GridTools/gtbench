#include "./ghex_comm.hpp"

#include <ghex/glue/gridtools/field.hpp>

namespace communication {

namespace ghex_comm {


std::function<void(storage_t &)>
comm_halo_exchanger(grid const &g, storage_t::storage_info_t const &sinfo)
{
    static std::unique_ptr<grid::comm_obj_type> comm_obj;
    if (!comm_obj)
    {
        comm_obj.reset( new grid::comm_obj_type{::gridtools::ghex::make_communication_object<grid::patterns_type>()} );
    }
    auto co_ptr = comm_obj.get();
    auto patterns_ptr = &g.patterns();
    const auto domain_id = g.domain_id(); 
    return [co_ptr,patterns_ptr,domain_id](const storage_t& storage) mutable
    {
        auto& co = *co_ptr;
        auto& patterns = *patterns_ptr;
    
        auto field = ::gridtools::ghex::wrap_gt_field(domain_id, storage);

        co.exchange(patterns(field)).wait();

    };
}

double comm_global_max(grid const &g, double t) {
  double max;
  MPI_Allreduce(&t, &max, 1, MPI_DOUBLE, MPI_SUM, g.mpi_comm());
  return max;
}

} // namespace ghex_comm

} // namespace communication

