#pragma once

#include "./computation.hpp"

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

  using p_data_in = gt::arg<0, storage_t>;
  using p_data_out = gt::arg<1, storage_t>;

  using p_dz = gt::arg<2, global_parameter_t>;
  using p_dt = gt::arg<3, global_parameter_t>;
  using p_coeff = gt::arg<4, global_parameter_t>;

  using p_a = gt::tmp_arg<5, storage_t>;
  using p_b = gt::tmp_arg<6, storage_t>;
  using p_c = gt::tmp_arg<7, storage_t>;
  using p_d = gt::tmp_arg<8, storage_t>;

  using p_alpha = gt::arg<9, storage_ij_t>;
  storage_ij_t alpha_;
  using p_beta = gt::arg<10, storage_ij_t>;
  storage_ij_t beta_;
  using p_gamma = gt::arg<11, storage_ij_t>;
  storage_ij_t gamma_;
  using p_fact = gt::arg<12, storage_ij_t>;
  storage_ij_t fact_;

  using p_z = gt::tmp_arg<13, storage_t>;
  using p_x = gt::tmp_arg<14, storage_t>;

  using p_k_size = gt::arg<15, global_parameter_int_t>;

public:
  vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
           real_t coeff);

  void operator()(storage_t &out, storage_t const &in, real_t dt);

private:
  gt::computation<p_data_in, p_data_out, p_dt> comp_;
};

} // namespace diffusion