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

#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

namespace verification {

using order_verification_result_t =
    std::tuple<std::vector<std::size_t>, std::vector<double>,
               std::vector<double>>;

template <class F>
order_verification_result_t order_verification(F &&f, std::size_t n_min,
                                               std::size_t n_max) {
  std::vector<std::size_t> ns;
  std::vector<double> errors;

  std::size_t n = n_min;
  while (n <= n_max) {
    ns.push_back(n);
    errors.push_back(f(n));
    n *= 2;
  }

  std::vector<double> orders;
  for (std::size_t i = 1; i < errors.size(); ++i)
    orders.push_back(std::log2(errors[i - 1] / errors[i]));

  return std::make_tuple(ns, errors, orders);
}

void print_order_verification_result(
    order_verification_result_t const &result) {
  auto const &ns = std::get<0>(result);
  auto const &errors = std::get<1>(result);
  auto const &orders = std::get<2>(result);

  const auto initial_flags = std::cout.flags();
  const auto initial_precision = std::cout.precision();

  std::cout << std::left << std::setw(15) << "Resolution"
            << " " << std::setw(15) << "Error"
            << " " << std::setw(15) << "Order" << std::endl;
  for (std::size_t i = 0; i < ns.size(); ++i) {
    std::cout << std::setw(15) << ns[i] << " ";
    std::cout << std::setw(15) << std::setprecision(5) << std::scientific
              << errors[i];
    if (i > 0)
      std::cout << " " << std::setw(15) << std::setprecision(2) << std::fixed
                << orders[i - 1];
    std::cout << std::endl;
  }

  std::cout.flags(initial_flags);
  std::cout.precision(initial_precision);
}

bool check_order(order_verification_result_t const &result,
                 real_t expected_order, real_t atol, real_t rtol) {
  auto const &orders = std::get<2>(result);
  for (auto order : orders) {
    if (std::abs(order - expected_order) <=
        atol + rtol * std::abs(expected_order))
      return true;
  }
  return false;
}

} // namespace verification
