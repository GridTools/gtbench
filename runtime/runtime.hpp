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

#include "../common/options.hpp"
#include "../common/types.hpp"
#include "./computation.hpp"
#include "./discrete_analytical.hpp"
#include "./result.hpp"

namespace runtime {

template <class RuntimeTag>
void runtime_register_options(RuntimeTag, options &) {}

template <class RuntimeTag>
RuntimeTag runtime_init(RuntimeTag, options_values const &) {
  return {};
}

template <class RuntimeTag>
void register_options(RuntimeTag, options &options) {
  return runtime_register_options(RuntimeTag{}, options);
}

template <class RuntimeTag>
auto init(RuntimeTag, options_values const &options) {
  return runtime_init(RuntimeTag{}, options);
}

template <class Runtime, class Analytical, class Stepper>
result solve(Runtime &&rt, Analytical &&analytical, Stepper &&stepper,
             vec<std::size_t, 3> const &global_resolution, real_t tmax,
             real_t dt) {
  return runtime_solve(
      std::forward<Runtime>(rt), std::forward<Analytical>(analytical),
      std::forward<Stepper>(stepper), global_resolution, tmax, dt);
}

} // namespace runtime