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

#include "./io.hpp"

namespace io {
namespace numpy {

class time_series final : public io::time_series {
public:
  using io::time_series::time_series;
  void write_step(real_t time, numerics::solver_state const &state) override;
};

} // namespace numpy
} // namespace io