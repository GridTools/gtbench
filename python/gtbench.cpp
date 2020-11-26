#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <gtbench/common/types.hpp>
#include <gtbench/numerics/solver.hpp>

namespace py = pybind11;

namespace {

gtbench::numerics::solver_state py2cpp_solver_state(py::object state) {
  auto resolution = state.attr("resolution");
  gtbench::vec<std::size_t, 3> cpp_resolution{
      resolution.attr("__getitem__")(0).cast<std::size_t>(),
      resolution.attr("__getitem__")(1).cast<std::size_t>(),
      resolution.attr("__getitem__")(2).cast<std::size_t>(),
  };
  auto delta = state.attr("delta");
  gtbench::vec<gtbench::real_t, 3> cpp_delta{
      delta.attr("__getitem__")(0).cast<gtbench::real_t>(),
      delta.attr("__getitem__")(1).cast<gtbench::real_t>(),
      delta.attr("__getitem__")(2).cast<gtbench::real_t>(),
  };
  gtbench::storage_t u = state.attr("u").cast<gtbench::storage_t>();
  gtbench::storage_t v = state.attr("v").cast<gtbench::storage_t>();
  gtbench::storage_t w = state.attr("w").cast<gtbench::storage_t>();
  gtbench::storage_t data = state.attr("data").cast<gtbench::storage_t>();
  gtbench::storage_t data1 = state.attr("data1").cast<gtbench::storage_t>();
  gtbench::storage_t data2 = state.attr("data2").cast<gtbench::storage_t>();
  return gtbench::numerics::solver_state(cpp_resolution, cpp_delta, u, v, w,
                                         data, data1, data2);
}

gtbench::numerics::exchange_t py2cpp_exchange(py::object exchange) {
  return
      [exchange](gtbench::storage_t &storage) { exchange(py::cast(storage)); };
}

void reassign_fields(gtbench::numerics::solver_state const &cpp_state,
                     py::object state) {
  state.attr("u") = py::cast(cpp_state.u);
  state.attr("v") = py::cast(cpp_state.v);
  state.attr("w") = py::cast(cpp_state.w);
  state.attr("data") = py::cast(cpp_state.data);
  state.attr("data1") = py::cast(cpp_state.data1);
  state.attr("data2") = py::cast(cpp_state.data2);
}

py::cpp_function py2cpp_stepper(gtbench::numerics::stepper_t &&cpp_stepper) {
  return py::cpp_function(
      [cpp_stepper = std::move(cpp_stepper)](py::object state,
                                             py::object exchange) {
        auto cpp_state = py2cpp_solver_state(state);
        auto cpp_exchange = py2cpp_exchange(exchange);
        auto cpp_step = cpp_stepper(cpp_state, cpp_exchange);
        return py::cpp_function(
            [cpp_step = std::move(cpp_step)](py::object state,
                                             gtbench::real_t dt) {
              auto cpp_state = py2cpp_solver_state(state);
              cpp_step(cpp_state, dt);
              reassign_fields(cpp_state, state);
            },
            py::arg("state"), py::arg("dt"));
      },
      py::arg("state"), py::arg("exchange"));
}

} // namespace

PYBIND11_MODULE(gtbenchpy, m) {
  m.doc() = "GTBench Python bindings";

  m.attr("halo") = gtbench::halo;
  m.attr("dtype") =
      std::is_same<gtbench::real_t, float>() ? "float32" : "float64";

  py::class_<typename gtbench::storage_t::element_type, gtbench::storage_t>(
      m, "Storage", py::buffer_protocol())
      .def(py::init([](py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<gtbench::real_t>::format())
          throw std::runtime_error("Incompatible dtype");
        if (info.ndim != 3)
          throw std::runtime_error("Wrong number of dimensions");

        const std::ptrdiff_t si = info.strides[0] / sizeof(gtbench::real_t);
        const std::ptrdiff_t sj = info.strides[1] / sizeof(gtbench::real_t);
        const std::ptrdiff_t sk = info.strides[2] / sizeof(gtbench::real_t);
        const gtbench::real_t *data =
            reinterpret_cast<gtbench::real_t *>(info.ptr);

        auto storage = gtbench::storage_builder(
            {std::size_t(info.shape[0] - 2 * gtbench::halo),
             std::size_t(info.shape[1] - 2 * gtbench::halo),
             std::size_t(info.shape[2] - 1)})();

        auto view = storage->host_view();
        const long ni = view.lengths()[0];
        const long nj = view.lengths()[1];
        const long nk = view.lengths()[2];
#pragma omp parallel for collapse(3)
        for (long k = 0; k < nk; ++k)
          for (long j = 0; j < nj; ++j)
            for (long i = 0; i < ni; ++i)
              view(i, j, k) = data[i * si + j * sj + k * sk];

        return storage;
      }))
      .def_buffer([](typename gtbench::storage_t::element_type &storage)
                      -> py::buffer_info {
        return py::buffer_info(
            storage.get_target_ptr(), sizeof(gtbench::real_t),
            py::format_descriptor<gtbench::real_t>::format(), 3,
            {storage.lengths()[0], storage.lengths()[1], storage.lengths()[2]},
            {storage.strides()[0] * sizeof(gtbench::real_t),
             storage.strides()[1] * sizeof(gtbench::real_t),
             storage.strides()[2] * sizeof(gtbench::real_t)});
      })
#ifdef GTBENCH_BACKEND_GPU
      .def_property_readonly(
          "__cuda_array_interface__",
          [](typename gtbench::storage_t::element_type &self) {
            iface = py::dict();
            iface["shape"] = py::make_tuple(
                self.lengths()[0], self.lengths()[1], self.lengths()[2]);
            iface["typestr"] = "<f" + std::itos(sizeof(gtbench::real_t));
            iface["data"] = py::make_tuple(self.get_target_ptr(), false);
            iface["version"] = 2;
            iface["strides"] = py::make_tuple(
                self.strides()[0], self.strides()[1], self.strides()[2]);
            return iface;
          })
#endif
      .def("__getitem__",
           [](typename gtbench::storage_t::element_type &self,
              py::object slice) {
#ifdef GTBENCH_BACKEND_GPU
             py::module_ cp = py::module_::import("cupy");
             return cp.attr("asarray")(py::cast(self))
                 .attr("__getitem__")(slice);
#else
             return py::array(py::cast(self)).attr("__getitem__")(slice);
#endif
           })
      .def("__setitem__", [](typename gtbench::storage_t::element_type &self,
                             py::object slice, py::object value) {
#ifdef GTBENCH_BACKEND_GPU
        py::module_ cp = py::module_::import("cupy");
        cp.attr("asarray")(py::cast(self)).attr("__setitem__")(slice, value);
#else
        py::array(py::cast(self)).attr("__setitem__")(slice, value);
#endif
      });
  m.def("storage_from_array", [m](py::array_t<gtbench::real_t> array) {
    return m.attr("Storage")(array);
  });
  m.def(
      "array_from_storage",
      [](typename gtbench::storage_t::element_type &storage) {
        gtbench::real_t const *storage_ptr = storage.get_const_host_ptr();
        std::size_t size = storage.length() * sizeof(gtbench::real_t);
        gtbench::real_t *ptr = (gtbench::real_t *)std::malloc(size);
        std::memcpy(ptr, storage_ptr, size);

        py::capsule capsule(ptr, [](void *ptr) { std::free(ptr); });
        return py::array_t<gtbench::real_t>(
            {storage.lengths()[0], storage.lengths()[1], storage.lengths()[2]},
            {storage.strides()[0] * sizeof(gtbench::real_t),
             storage.strides()[1] * sizeof(gtbench::real_t),
             storage.strides()[2] * sizeof(gtbench::real_t)},
            ptr, capsule);
      });

  m.def("hdiff_stepper", [](gtbench::real_t diffusion_coeff) {
    auto cpp_stepper = gtbench::numerics::hdiff_stepper(diffusion_coeff);
    return py2cpp_stepper(std::move(cpp_stepper));
  });
  m.def("vdiff_stepper", [](gtbench::real_t diffusion_coeff) {
    auto cpp_stepper = gtbench::numerics::vdiff_stepper(diffusion_coeff);
    return py2cpp_stepper(std::move(cpp_stepper));
  });
  m.def("diff_stepper", [](gtbench::real_t diffusion_coeff) {
    auto cpp_stepper = gtbench::numerics::diff_stepper(diffusion_coeff);
    return py2cpp_stepper(std::move(cpp_stepper));
  });
  m.def("hadv_stepper", []() {
    auto cpp_stepper = gtbench::numerics::hadv_stepper();
    return py2cpp_stepper(std::move(cpp_stepper));
  });
  m.def("vadv_stepper", []() {
    auto cpp_stepper = gtbench::numerics::vadv_stepper();
    return py2cpp_stepper(std::move(cpp_stepper));
  });
  m.def("rkadv_stepper", []() {
    auto cpp_stepper = gtbench::numerics::rkadv_stepper();
    return py2cpp_stepper(std::move(cpp_stepper));
  });
  m.def("advdiff_stepper", [](gtbench::real_t diffusion_coeff) {
    auto cpp_stepper = gtbench::numerics::advdiff_stepper(diffusion_coeff);
    return py2cpp_stepper(std::move(cpp_stepper));
  });
}
