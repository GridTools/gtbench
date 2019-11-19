#pragma once

#include "common.hpp"

struct periodic_boundary {
  template <gt::sign I, gt::sign J, gt::sign K, typename DataField>
  GT_FUNCTION void operator()(gt::direction<I, J, K>, DataField &data,
                              gt::uint_t i, gt::uint_t j, gt::uint_t k) const {
    auto const &si = data.storage_info();
    data(i, j, k) = data(
        (i + si.template length<0>() - halo) % si.template length<0>() + halo,
        (j + si.template length<1>() - halo) % si.template length<1>() + halo,
        k);
  }
};