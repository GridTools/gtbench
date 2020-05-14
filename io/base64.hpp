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

#include <cstdint>
#include <ostream>

namespace io {
class base64_encoder {
  std::ostream &out;
  std::uint32_t buffer = 0;
  unsigned i = 0;

public:
  base64_encoder(std::ostream &out) : out(out) {}
  ~base64_encoder() { flush(); }

  base64_encoder &write(const void *data, std::size_t size) {
    auto bytes = (const std::uint8_t *)data;
    for (; size > 0; --size) {
      buffer |= *(bytes++) << 8 * (2 - i++);
      if (i == 3)
        flush();
    }
    return *this;
  }

  base64_encoder &flush() {
    if (i != 0) {
      static constexpr const char *chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                           "abcdefghijklmnopqrstuvwxyz"
                                           "0123456789+/";
      out << chars[(buffer >> 18) & 0x3f] << chars[(buffer >> 12) & 0x3f]
          << (i > 1 ? chars[(buffer >> 6) & 0x3f] : '=')
          << (i > 2 ? chars[buffer & 0x3f] : '=');
      buffer = 0;
      i = 0;
    }
    return *this;
  }
};
} // namespace io
