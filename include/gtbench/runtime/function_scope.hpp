/*
 * gtbench
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>

namespace gtbench {
namespace runtime {

class function_scope {
public:
  function_scope() {}
  function_scope(std::function<void()> const &init,
                 std::function<void()> const &finalize)
      : m_finalize(finalize) {
    init();
  }
  function_scope(function_scope const &) = delete;
  function_scope(function_scope &&other) = default;
  ~function_scope() {
    if (m_finalize)
      m_finalize();
  }

  function_scope &operator=(function_scope const &) = delete;
  function_scope &operator=(function_scope &&) = default;

private:
  std::function<void()> m_finalize;
};

} // namespace runtime
} // namespace gtbench
