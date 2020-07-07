/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "./computation.hpp"

namespace numerics {
namespace diffusion {

class horizontal {
  using p_out = gt::arg<0, storage_t>;
  using p_in = gt::arg<1, storage_t>;
  using p_dx = gt::arg<2, global_parameter_t>;
  using p_dy = gt::arg<3, global_parameter_t>;
  using p_dt = gt::arg<4, global_parameter_t>;
  using p_coeff = gt::arg<5, global_parameter_t>;

public:
  horizontal(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
             real_t coeff);

  void operator()(storage_t &out, storage_t const &in, real_t dt);

private:
  gt::computation<p_out, p_in, p_dt> comp_;
};

class vertical {
  storage_ij_t::storage_info_t sinfo_ij_;
  storage_t::storage_info_t sinfo_;

  using p_data_in = gt::arg<0, storage_t>;
  using p_data_in_uncached = gt::arg<1, storage_t>;
  using p_data_out = gt::arg<2, storage_t>;

  using p_dz = gt::arg<3, global_parameter_t>;
  using p_dt = gt::arg<4, global_parameter_t>;
  using p_coeff = gt::arg<5, global_parameter_t>;

  using p_c = gt::tmp_arg<6, storage_t>;
  using p_d = gt::arg<7, storage_t>;
  using p_d_uncached = gt::arg<8, storage_t>;
  storage_t d_;
  using p_c2 = gt::tmp_arg<9, storage_t>;
  using p_d2 = gt::arg<10, storage_t>;
  using p_d2_uncached = gt::arg<11, storage_t>;
  storage_t d2_;

  using p_fact = gt::arg<12, storage_ij_t>;
  storage_ij_t fact_;

  using p_k_size = gt::arg<13, global_parameter_int_t>;

public:
  vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
           real_t coeff);

  void operator()(storage_t &out, storage_t const &in, real_t dt);

private:
  gt::computation<p_data_in, p_data_in_uncached, p_dt> comp1_;
  gt::computation<p_data_out> comp2_;
};

} // namespace diffusion
} // namespace numerics
