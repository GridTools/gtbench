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

#include <cstdint>
#include <ostream>

namespace gtbench {
namespace io {
class base64_encoder {
  std::ostream &m_out;
  std::uint32_t m_buffer = 0;
  unsigned m_i = 0;

public:
  base64_encoder(std::ostream &out) : m_out(out) {}
  ~base64_encoder() { flush(); }

  base64_encoder &write(const void *data, std::size_t size) {
    auto bytes = (const std::uint8_t *)data;
    for (; size > 0; --size) {
      m_buffer |= *(bytes++) << 8 * (2 - m_i++);
      if (m_i == 3)
        flush();
    }
    return *this;
  }

  base64_encoder &flush() {
    if (m_i != 0) {
      static constexpr const char *chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                           "abcdefghijklmnopqrstuvwxyz"
                                           "0123456789+/";
      m_out << chars[(m_buffer >> 18) & 0x3f] << chars[(m_buffer >> 12) & 0x3f]
            << (m_i > 1 ? chars[(m_buffer >> 6) & 0x3f] : '=')
            << (m_i > 2 ? chars[m_buffer & 0x3f] : '=');
      m_buffer = 0;
      m_i = 0;
    }
    return *this;
  }
};
} // namespace io
} // namespace gtbench
