/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./options.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace options_impl {

void abort(std::string const &message) {
  std::cerr << "command line parsing error: " << message << std::endl;
  std::exit(1);
}

} // namespace options_impl

options_values options::parse(int argc, char **argv) const {
  options_values parsed;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);

    if (arg == "-h" || arg == "--help") {
      print_help(argv[0]);
      std::exit(0);
    }

    if (arg[0] == '-' && arg[1] == '-') {
      std::string name = arg.substr(2);
      auto opt = std::find_if(m_options.begin(), m_options.end(),
                              [&](option const &o) { return o.name == name; });
      if (opt == m_options.end())
        options_impl::abort("unkown option: '" + arg + "'");

      if (parsed.m_map.find(name) != parsed.m_map.end())
        options_impl::abort("multiple occurences of '" + arg + "'");

      std::vector<std::string> values;
      for (int j = 0; j < opt->nargs; ++j, ++i) {
        if (i + 1 >= argc)
          options_impl::abort(
              "unexpected end of arguments while parsing args for '" + arg +
              "'");
        std::string value(argv[i + 1]);
        if (value[0] == '-' && value[1] == '-')
          options_impl::abort("expected argument for option '" + arg +
                              "', found '" + value + "'");
        values.push_back(value);
      }

      parsed.m_map[name] = values;
    } else {
      options_impl::abort("unexpected token: '" + arg +
                          "' (too many arguments provided?)");
    }
  }

  for (auto const &opt : m_options) {
    if (!opt.default_values.empty() &&
        parsed.m_map.find(opt.name) == parsed.m_map.end()) {
      parsed.m_map[opt.name] = opt.default_values;
    }
  }

  return parsed;
}

void options::print_help(std::string const &command) const {
  std::cout << "usage: " << command << " [options...]" << std::endl;

  std::size_t max_opt_len = 4;
  for (option const &opt : m_options) {
    max_opt_len = std::max(opt.name.size() + opt.variable.size(), max_opt_len);
  }

  auto print =
      [max_opt_len](std::string const &opt, std::string const &descr,
                    std::vector<std::string> const &default_values = {}) {
        std::cout << "    " << std::left << std::setw(max_opt_len + 3) << opt
                  << " " << descr;
        if (!default_values.empty()) {
          std::cout << " (default:";
          for (auto const &default_value : default_values)
            std::cout << " " << default_value;
          std::cout << ")";
        }
        std::cout << std::endl;
      };

  print("--help", "print this help message and exit");

  for (option const &opt : m_options) {
    print("--" + opt.name + " " + opt.variable, opt.description,
          opt.default_values);
  }
}