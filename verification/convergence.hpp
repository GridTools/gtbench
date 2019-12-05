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
  std::vector<std::size_t> ns;
  std::vector<double> errors, orders;
  std::tie(ns, errors, orders) = result;

  std::cout << std::setw(5) << "N" << std::setw(20) << "Error" << std::setw(20)
            << "Order" << std::endl;
  for (std::size_t i = 0; i < ns.size(); ++i) {
    std::cout << std::setw(5) << ns[i] << " " << std::setw(20) << errors[i];
    if (i > 0)
      std::cout << std::setw(20) << orders[i - 1];
    std::cout << std::endl;
  }
}

} // namespace verification