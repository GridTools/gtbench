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

#include <memory>
#include <string>

#include "../common/types.hpp"
#include "../numerics/solver.hpp"

namespace io {

class time_series {
private:
  std::string m_filename;
  vec<std::size_t, 3> m_global_resolution, m_local_resolution, m_local_offset;

public:
  time_series(std::string const &filename,
              vec<std::size_t, 3> const &global_resolution,
              vec<std::size_t, 3> const &local_resolution,
              vec<std::size_t, 3> const &local_offset);
  virtual ~time_series() {}

  std::string const &filename() const { return m_filename; }
  vec<std::size_t, 3> const &global_resolution() const {
    return m_global_resolution;
  }
  vec<std::size_t, 3> const &local_resolution() const {
    return m_local_resolution;
  }
  vec<std::size_t, 3> const &local_offset() const { return m_local_offset; }

  virtual void write_step(real_t time, numerics::solver_state const &state) = 0;

protected:
  std::size_t rank(vec<std::size_t, 3> const &local_offset) const;
  std::size_t rank() const;
  std::size_t ranks() const;
};

std::shared_ptr<time_series>
time_series_from_filename(std::string const &filename,
                          vec<std::size_t, 3> const &global_resolution,
                          vec<std::size_t, 3> const &local_resolution,
                          vec<std::size_t, 3> const &local_offset);
} // namespace io