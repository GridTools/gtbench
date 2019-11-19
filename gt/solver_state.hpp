#pragma once

#include "common.hpp"

struct solver_state {
  template <class DataInit, class UInit, class VInit, class WInit>
  solver_state(std::size_t resolution_x, std::size_t resolution_y,
               std::size_t resolution_z, DataInit &&data_init, UInit &&u_init,
               VInit &&v_init, WInit &&w_init)
      : sinfo(resolution_x + 2 * halo, resolution_y + 2 * halo, resolution_z),
        data(sinfo, std::forward<DataInit>(data_init), "data"),
        u(sinfo, std::forward<UInit>(u_init), "u"),
        v(sinfo, std::forward<VInit>(v_init), "v"),
        w(sinfo, std::forward<WInit>(w_init), "w"), tmp(sinfo, "tmp") {}

  storage_t::storage_info_t sinfo;
  storage_t data, u, v, w, tmp;
};
