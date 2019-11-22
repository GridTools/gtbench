#pragma once

//#include <gridtools/storage/data_store.hpp>
#include <gridtools/storage/storage_facility.hpp>

namespace gt = gridtools;

using real_t = double;

static constexpr gt::int_t halo = 3;
static constexpr gt::int_t huge_offset = 10000;

using backend_t = gt::backend::x86;
using storage_tr = gt::storage_traits<backend_t>;
using storage_info_ijk_t =
    storage_tr::storage_info_t<0, 3, gt::halo<halo, halo, 0>>;
using storage_info_ij_t =
    storage_tr::special_storage_info_t<3, gt::selector<1, 1, 0>,
                                       gt::halo<halo, halo, 0>>;
using storage_t = storage_tr::data_store_t<real_t, storage_info_ijk_t>;
using storage_ij_t = storage_tr::data_store_t<real_t, storage_info_ij_t>;
