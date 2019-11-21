#pragma once

#include "../common.hpp"

namespace advection {

class horizontal {
  using p_out = gt::arg<0, storage_t>;
  using p_in = gt::arg<1, storage_t>;
  using p_u = gt::arg<2, storage_t>;
  using p_v = gt::arg<3, storage_t>;
  using p_dx = gt::arg<4, global_parameter_t>;
  using p_dy = gt::arg<5, global_parameter_t>;
  using p_dt = gt::arg<6, global_parameter_t>;

public:
  horizontal(grid_t const &grid, real_t dx, real_t dy);

  void operator()(storage_t &out, storage_t const &in, storage_t const &u,
                  storage_t const &v, real_t dt);

private:
  gt::computation<p_out, p_in, p_u, p_v, p_dt> comp_;
};

class vertical {
  storage_ij_t::storage_info_t sinfo_ij_;

  using p_data_in = gt::arg<0, storage_t>;
  using p_data_out = gt::arg<1, storage_t>;

  using p_dz = gt::arg<2, global_parameter_t>;
  using p_dt = gt::arg<3, global_parameter_t>;

  using p_a = gt::tmp_arg<4, storage_t>;
  using p_b = gt::tmp_arg<5, storage_t>;
  using p_c = gt::tmp_arg<6, storage_t>;
  using p_d = gt::tmp_arg<7, storage_t>;

  using p_alpha = gt::arg<8, storage_ij_t>;
  storage_ij_t alpha_;
  using p_beta = gt::arg<9, storage_ij_t>;
  storage_ij_t beta_;
  using p_gamma = gt::arg<10, storage_ij_t>;
  storage_ij_t gamma_;
  using p_fact = gt::arg<11, storage_ij_t>;
  storage_ij_t fact_;

  using p_z = gt::tmp_arg<12, storage_t>;
  using p_x = gt::tmp_arg<13, storage_t>;

  using p_w = gt::arg<14, storage_t>;

  using p_k_size = gt::arg<15, global_parameter_int_t>;

public:
  vertical(grid_t const &grid, real_t dz);

  void operator()(storage_t &out, storage_t const &in, storage_t const &w,
                  real_t dt);

private:
  gt::computation<p_data_in, p_data_out, p_w, p_dt> comp_;
};

class runge_kutta_step {
  storage_ij_t::storage_info_t sinfo_ij_;

  using p_data_in = gt::arg<0, storage_t>;
  using p_data_in0 = gt::arg<1, storage_t>;
  using p_data_out = gt::arg<2, storage_t>;

  using p_dx = gt::arg<3, global_parameter_t>;
  using p_dy = gt::arg<4, global_parameter_t>;
  using p_dz = gt::arg<5, global_parameter_t>;
  using p_dt = gt::arg<6, global_parameter_t>;

  using p_a = gt::tmp_arg<7, storage_t>;
  using p_b = gt::tmp_arg<8, storage_t>;
  using p_c = gt::tmp_arg<9, storage_t>;
  using p_d = gt::tmp_arg<10, storage_t>;

  using p_alpha = gt::arg<11, storage_ij_t>;
  storage_ij_t alpha_;
  using p_beta = gt::arg<12, storage_ij_t>;
  storage_ij_t beta_;
  using p_gamma = gt::arg<13, storage_ij_t>;
  storage_ij_t gamma_;
  using p_fact = gt::arg<14, storage_ij_t>;
  storage_ij_t fact_;

  using p_z = gt::tmp_arg<15, storage_t>;
  using p_x = gt::tmp_arg<16, storage_t>;

  using p_u = gt::arg<17, storage_t>;
  using p_v = gt::arg<18, storage_t>;
  using p_w = gt::arg<19, storage_t>;

  using p_k_size = gt::arg<20, global_parameter_int_t>;

public:
  runge_kutta_step(grid_t const &grid, real_t dx, real_t dy, real_t dz);

  void operator()(storage_t &out, storage_t const &in, storage_t const &in0,
                  storage_t const &u, storage_t const &v, storage_t const &w,
                  real_t dt);

private:
  gt::computation<p_data_out, p_data_in, p_data_in0, p_u, p_v, p_w, p_dt> comp_;
};

} // namespace advection