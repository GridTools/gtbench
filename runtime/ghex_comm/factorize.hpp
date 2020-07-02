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

#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

namespace runtime {

namespace ghex_comm_impl {

template <typename I> std::vector<I> factorize(I n) {
  std::vector<I> result;
  for (I i = 2; i <= n; ++i) {
    if ((n % i) == 0) {
      result.push_back(i);
      n /= i;
      --i;
    }
  }
  return result;
}
template <std::size_t N, typename I, typename J>
void partition_factors(const std::vector<I> &f, const std::array<J, N> &weights,
                       std::size_t i, std::array<I, N> &f_d, double &min_var,
                       std::array<I, N> &f_d_min) {
  if (i >= f.size()) {
    // end of recursion
    auto min_element = ((double)weights[0]) / f_d[0];
    auto max_element = min_element;
    for (std::size_t j = 1; j < N; ++j) {
      const auto sample = ((double)weights[j]) / f_d[j];
      min_element = std::min(min_element, sample);
      max_element = std::max(max_element, sample);
    }
    double var = max_element - min_element;
    if (var < min_var) {
      min_var = var;
      f_d_min = f_d;
    }
  } else {
    for (std::size_t j = 0; j < N; ++j) {
      f_d[j] *= f[i];
      partition_factors(f, weights, i + 1, f_d, min_var, f_d_min);
      f_d[j] /= f[i];
    }
  }
}
template <std::size_t N, typename I, typename J>
std::array<I, N> partition_factors(const std::vector<I> &f,
                                   const std::array<J, N> &weights) {
  std::array<I, N> f_d, f_d_min;
  f_d.fill(I{1});
  double min_var = *std::max_element(weights.begin(), weights.end());
  partition_factors(f, weights, 0, f_d, min_var, f_d_min);
  return f_d_min;
}

template <typename I, typename J, typename K, std::size_t N>
typename std::enable_if<std::is_integral<I>::value &&
                            std::is_integral<J>::value &&
                            std::is_integral<K>::value,
                        std::array<std::vector<J>, N>>::type
divide_domain(I n, const std::array<J, N> &sizes,
              const std::array<K, N> &factors) {
  // compute the sub-domain size per dimension
  std::array<double, N> dx;
  for (std::size_t i = 0; i < N; ++i) {
    dx[i] = sizes[i] / (double)factors[i];
    while (dx[i] * factors[i] < sizes[i])
      dx[i] += std::numeric_limits<double>::epsilon() * sizes[i];
  }
  // make a vector of sub-domains per dimension
  std::array<std::vector<J>, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i].resize(factors[i], 1);
    for (I j = 0; j < factors[i]; ++j) {
      const I l = j * dx[i];
      const I u = (j + 1) * dx[i];
      result[i][j] = u - l;
    }
    std::sort(result[i].begin(), result[i].end());
  }
  return result;
}

} // namespace ghex_comm_impl
} // namespace runtime
