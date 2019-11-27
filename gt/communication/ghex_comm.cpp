#include "./ghex_comm.hpp"

#include <ghex/glue/gridtools/field.hpp>

namespace communication {

namespace ghex_comm {


std::function<void(storage_t &)>
comm_halo_exchanger(grid const &grid, storage_t::storage_info_t const &sinfo)
{
    static std::unique_ptr<typename grid::comm_obj_type> comm_obj;
    if (!comm_obj)
    {
        comm_obj.reset( new typename grid::comm_obj_type{::gridtools::ghex::make_communication_object<typename grid::patterns_type>()} );
    }
    auto co_ptr = comm_obj.get();
    auto patterns_ptr = &grid.patterns();

    using domain_id_type    = typename local_domain::domain_id_type;
    using arch_t            = typename ::gridtools::ghex::_impl::get_arch<storage_t>::type;
    using value_t           = typename storage_t::data_t;
    using layout_t          = typename storage_t::storage_info_t::layout_t;
    using integer_seq       = typename ::gridtools::_impl::get_unmasked_layout_map<layout_t>::integer_seq;
    using uint_t            = decltype(layout_t::masked_length);
    using dimension         = std::integral_constant<uint_t, layout_t::masked_length>;
    using halo_t            = typename storage_t::storage_info_t::halo_t;
    using field_type        = typename ::gridtools::ghex::_impl::get_simple_field_wrapper_type<value_t, arch_t, local_domain, integer_seq>::type;

    const auto& domain_id = grid.domain_id(); 
    const auto& extents = sinfo.total_lengths();
    const auto& origin  = ::gridtools::_impl::get_begin<halo_t, uint_t>(std::make_index_sequence<dimension::value>());
    auto strides       = sinfo.strides();
    for (unsigned int i=0u; i<dimension::value; ++i)
        strides[i] *= sizeof(value_t);
    typename ::gridtools::ghex::arch_traits<arch_t>::device_id_type device_id = 0;
    field_type field(domain_id, (value_t*)0, origin, extents, strides, device_id);

    return [co_ptr,patterns_ptr,field](const storage_t& storage) mutable
    {
        auto& co = *co_ptr;
        auto& patterns = *patterns_ptr;
        field.set_data(storage.get_storage_ptr()->get_target_ptr());

        co.exchange(patterns(field)).wait();

    };
}

double comm_global_sum(grid const &grid, double t) {
  MPI_Allreduce(&t, &t, 1, MPI_DOUBLE, MPI_SUM, grid.mpi_comm());
  return t;
}

} // namespace ghex_comm

} // namespace communication

