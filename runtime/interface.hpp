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

#include "../numerics/solver.hpp"
#include "./discrete_analytical.hpp"

namespace runtime {

namespace impl {
template <class F> void init_field(storage_t &storage, F f, real_t t) {
  auto view = gt::make_host_view(storage);
  const long ni = storage.total_length<0>();
  const long nj = storage.total_length<1>();
  const long nk = storage.total_length<2>();

#pragma omp parallel for collapse(3)
  for (long k = 0; k < nk; ++k)
    for (long j = 0; j < nj; ++j)
      for (long i = 0; i < ni; ++i)
        view(i, j, k) = f({i, j, k}, t);

  storage.sync();
}

template <class F>
double compute_field_error(storage_t const &storage, F f, real_t t) {
  storage.sync();
  auto view = gt::make_host_view<gt::access_mode::read_only>(storage);
  const long ni = storage.total_length<0>();
  const long nj = storage.total_length<1>();
  const long nk = storage.total_length<2>();

  double error = 0.0;
#pragma omp parallel for collapse(3) reduction(max : error)
  for (long k = 0; k < nk - 1; ++k)
    for (long j = halo; j < nj - halo; ++j)
      for (long i = halo; i < ni - halo; ++i)
        error =
            std::max(error, double(std::abs(view(i, j, k) - f({i, j, k}, t))));
  return error;
}
} // namespace impl

template <class Discrete>
numerics::solver_state init_state(Discrete const &discrete, real_t t = 0_r) {
  vec<std::size_t, 3> local_resolution =
      discrete_analytical::local_resolution(discrete);
  vec<real_t, 3> delta = discrete_analytical::delta(discrete);
  numerics::solver_state state(local_resolution, delta);

  impl::init_field(state.data, discrete_analytical::data(discrete), t);
  impl::init_field(state.u, discrete_analytical::u(discrete), t);
  impl::init_field(state.v, discrete_analytical::v(discrete), t);
  impl::init_field(state.w, discrete_analytical::w(discrete), t);

  return state;
}

template <class Discrete>
double compute_error(numerics::solver_state const &state,
                     Discrete const &discrete, real_t t) {
  return impl::compute_field_error(state.data,
                                   discrete_analytical::data(discrete), t);
}

} // namespace runtime