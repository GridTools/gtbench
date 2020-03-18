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

#include <cxxopts.hpp>

#include "../common/types.hpp"

namespace runtime {

struct result {
  double error;
  double time;
};

template <class World>
void runtime_register_options(World const &, cxxopts::Options &) {}

template <class World>
World runtime_init(World const &world, cxxopts::ParseResult const &) {
  return world;
}

template <class Runtime>
void register_options(Runtime &&rt, cxxopts::Options &options) {
  return runtime_register_options(std::forward<Runtime>(rt), options);
}

template <class Runtime>
decltype(auto) init(Runtime &&rt, cxxopts::ParseResult const &options) {
  return runtime_init(std::forward<Runtime>(rt), options);
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