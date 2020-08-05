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

#include <gridtools/boundaries/boundary.hpp>

namespace runtime {
namespace single_node_impl {

void runtime_register_options(single_node, options &options) {
  options("output", "optional data output", "FILE");
}

runtime runtime_init(single_node, options_values const &options) {
  return {options.get_or<std::string>("output", "")};
}

struct periodic_boundary {
  template <gt::boundaries::sign I, gt::boundaries::sign J,
            gt::boundaries::sign K, typename DataField>
  GT_FUNCTION void operator()(gt::boundaries::direction<I, J, K>,
                              DataField &data, gt::uint_t i, gt::uint_t j,
                              gt::uint_t k) const {
    auto const inner_size_i = data.lengths()[0] - 2 * halo;
    auto const inner_size_j = data.lengths()[1] - 2 * halo;
    data(i, j, k) = data((i + inner_size_i - halo) % inner_size_i + halo,
                         (j + inner_size_j - halo) % inner_size_j + halo, k);
  }
};

numerics::exchange_t exchange_func(vec<std::size_t, 3> const &resolution) {
  gt::uint_t nx = resolution.x;
  gt::uint_t ny = resolution.y;
  gt::uint_t nz = resolution.z;
  const gt::array<gt::halo_descriptor, 3> halos{
      {{halo, halo, halo, halo + nx - 1, halo + nx + halo},
       {halo, halo, halo, halo + ny - 1, halo + ny + halo},
       {0, 0, 0, nz - 1, nz}}};
  gt::boundaries::boundary<periodic_boundary, backend_t> boundary(
      halos, periodic_boundary());

  return [boundary = std::move(boundary)](storage_t &storage) {
    boundary.apply(storage);
  };
}

} // namespace single_node_impl
} // namespace runtime
