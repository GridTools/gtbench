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

#include "../common/types.hpp"

namespace runtime {

struct result {
  double error;
  double time;
};

template <class Runtime, class Analytical, class Stepper>
result solve(Runtime &&rt, Analytical &&analytical, Stepper &&stepper,
             vec<std::size_t, 3> const &global_resolution, real_t tmax,
             real_t dt) {
  return runtime_solve(
      std::forward<Runtime>(rt), std::forward<Analytical>(analytical),
      std::forward<Stepper>(stepper), global_resolution, tmax, dt);
}

} // namespace runtime