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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace options_impl {

void abort(std::string const &message);

} // namespace options_impl

class options_values {
  template <class T> struct type {};

public:
  bool has(std::string const &name) const {
    return m_map.find(name) != m_map.end();
  }

  template <class T> T get(std::string const &name) const {
    auto value = m_map.find(name);
    if (value == m_map.end())
      options_impl::abort("no value found for option '" + name + "'");
    return parse(value->second, type<T>());
  }

  template <class T>
  T get(std::string const &name, T const &default_value) const {
    auto value = m_map.find(name);
    if (value == m_map.end())
      return default_value;
    return parse(value->second, type<T>());
  }

  operator bool() const { return !m_map.empty(); }

private:
  template <class T> static T parse_value(std::string const &value) {
    std::istringstream value_stream(value);
    T result;
    value_stream >> result;
    return result;
  }

  template <class T>
  static T parse(std::vector<std::string> const &values, type<T>) {
    if (values.size() != 1)
      options_impl::abort("wrong number of arguments requested");
    return parse_value<T>(values.front());
  }

  template <class T, std::size_t N>
  static std::array<T, N> parse(std::vector<std::string> const &values,
                                type<std::array<T, N>>) {
    if (values.size() != N)
      options_impl::abort("wrong number of arguments requested");
    std::array<T, N> result;
    for (std::size_t i = 0; i < N; ++i)
      result[i] = parse_value<T>(values[i]);
    return result;
  }

  std::map<std::string, std::vector<std::string>> m_map;

  friend class options;
};

class options {
  struct option {
    std::string name;
    std::string description;
    std::string variable;
    std::vector<std::string> default_values;
    std::size_t nargs;
  };

public:
  options_values parse(int argc, char **argv) const;

  options &operator()(std::string const &name, std::string const &description,
                      std::string const &variable, std::size_t nargs = 1) {
    return add(name, description, variable, {}, nargs);
  }

  template <class T>
  options &operator()(std::string const &name, std::string const &description,
                      std::string const &variable,
                      std::initializer_list<T> default_values) {
    std::vector<std::string> default_str_values;
    for (auto const &value : default_values) {
      std::ostringstream value_stream;
      value_stream << value;
      default_str_values.push_back(value_stream.str());
    }
    return add(name, description, variable, default_str_values,
               default_str_values.size());
  }

private:
  options &add(std::string const &name, std::string const &description,
               std::string const &variable,
               std::vector<std::string> const &default_values,
               std::size_t nargs) {
    assert(default_values.size() == 0 || default_values.size() == nargs);
    m_options.push_back({name, description, variable, default_values, nargs});
    return *this;
  }

  std::string help_message(std::string const &command) const;

  std::vector<option> m_options;
};
