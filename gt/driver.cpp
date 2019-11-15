#include <fstream>
#include <iostream>

#include "common.hpp"
#include "diffusion.hpp"

template <typename T> void dump(std::ostream &os, T const &storage) {
  for (auto const &l : storage.total_lengths())
    os << l << " ";
  os << '\n';

  auto v = gt::make_host_view(storage);
  for (int i0 = 0; i0 < storage.template total_length<2>(); ++i0) {
    for (int i1 = 0; i1 < storage.template total_length<1>(); ++i1) {
      for (int i2 = 0; i2 < storage.template total_length<0>(); ++i2)
        os << v(i2, i1, i0) << " ";
      os << '\n';
    }
    os << '\n';
  }
}

struct periodic_boundary {
  template <gt::sign I, gt::sign J, gt::sign K, typename DataField>
  GT_FUNCTION void operator()(gt::direction<I, J, K>, DataField &data,
                              gt::uint_t i, gt::uint_t j, gt::uint_t k) const {
    auto const &si = data.storage_info();
    data(i, j, k) =
        data((i + si.template length<0>() - halo_i) % si.template length<0>() +
                 halo_i,
             (j + si.template length<1>() - halo_j) % si.template length<1>() +
                 halo_j,
             (k + si.template length<2>() - halo_k) % si.template length<2>() +
                 halo_k);
  }
};

int main() {
  static constexpr int isize = 30, jsize = 30, ksize = 30;
  real_t const dx = 1, dy = 1, dz = 1, dt = 1;
  real_t const diffusion_coefficient = 0.2;

  storage_t::storage_info_t sinfo{isize + 2 * halo_i, jsize + 2 * halo_j,
                                  ksize + 2 * halo_k};
  storage_ij_t::storage_info_t sinfo_ij{isize + 2 * halo_i, jsize + 2 * halo_j,
                                        1};

  storage_t u{sinfo, real_t(1), "u"}, v{sinfo, real_t(1), "v"};
  storage_t w{sinfo, real_t(1), "w"};
  storage_t data_in{sinfo,
                    [](int i, int j, int k) {
                      return i > 5 && i < 8 && j > 5 && j < 8 && k > 1 && k < 8
                                 ? real_t(1)
                                 : real_t(0);
                    },
                    "data"};
  storage_t data_out{sinfo, "data2"};
  storage_t flux{sinfo, "flux"};

  gt::array<gt::halo_descriptor, 3> halos{
      {{halo_i, halo_i, halo_i, halo_i + isize - 1, halo_i + isize + halo_i},
       {halo_j, halo_j, halo_j, halo_j + jsize - 1, halo_j + jsize + halo_j},
       {halo_k, halo_k, halo_k, halo_k + ksize - 1, halo_k + ksize + halo_k}}};
  auto grid = gt::make_grid(halos[0], halos[1], axis_t{ksize + 2 * halo_k});
  diffusion::horizontal hdiff(grid, dx, dy, dt, diffusion_coefficient);
  diffusion::vertical vdiff(grid, dz, dt, diffusion_coefficient, sinfo_ij);

  gt::boundary<periodic_boundary, backend_t> boundary(halos,
                                                      periodic_boundary{});

  boundary.apply(data_in);

  for (int ts = 0; ts < 10; ++ts) {
    std::ofstream of{"out" + std::to_string(ts)};
    dump(of, data_in);

    hdiff(data_out, data_in);
    boundary.apply(data_out);
    std::swap(data_out, data_in);

    vdiff(data_out, data_in);
    boundary.apply(data_out);
    std::swap(data_out, data_in);
  }
}
