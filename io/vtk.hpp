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

#include <fstream>
#include <string>
#include <vector>

#include "./io.hpp"

namespace io {
namespace vtk {

class time_series final : public io::time_series {
  std::vector<real_t> m_times;

  void write_pvd() const;
  void write_pvti(numerics::solver_state const &state) const;
  void write_vti(numerics::solver_state const &state) const;

public:
  using io::time_series::time_series;
  void write_step(real_t time, numerics::solver_state const &state) override;
};

} // namespace vtk

} // namespace io