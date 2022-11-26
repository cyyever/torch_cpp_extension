/*!
 * \file synced_tensor_dict.hpp
 *
 * \author cyy
 */

#pragma once
#include <torch/extension.h>

#include "../src/synced_tensor_dict.hpp"
namespace py = pybind11;
inline void define_torch_data_structure_extension(py::module_ &m) {
  using synced_tensor_dict = ::cyy::pytorch::synced_tensor_dict;
  auto sub_m = m.def_submodule("data_structure", "Contains data structures");
  py::class_<synced_tensor_dict>(sub_m, "SyncedTensorDict")
      .def(py::init<const std::string &>())
      .def("prefetch",
           static_cast<void (synced_tensor_dict::*)(
               const std::vector<std::string> &keys)>(
               &synced_tensor_dict::prefetch),
           py::call_guard<py::gil_scoped_release>())
      .def("set_in_memory_number", &synced_tensor_dict::set_in_memory_number,
           py::call_guard<py::gil_scoped_release>())
      .def("get_in_memory_number", &synced_tensor_dict::get_in_memory_number,
           py::call_guard<py::gil_scoped_release>())
      /* .def("set_storage_dir", &synced_tensor_dict::set_storage_dir, */
      /*      py::call_guard<py::gil_scoped_release>()) */
      .def("get_storage_dir", &synced_tensor_dict::get_storage_dir)
      .def("set_permanent_storage",
           &synced_tensor_dict::enable_permanent_storage,
           py::call_guard<py::gil_scoped_release>())
      .def("enable_permanent_storage",
           &synced_tensor_dict::enable_permanent_storage,
           py::call_guard<py::gil_scoped_release>())
      .def("disable_permanent_storage",
           &synced_tensor_dict::disable_permanent_storage,
           py::call_guard<py::gil_scoped_release>())
      .def("set_wait_flush_ratio", &synced_tensor_dict::set_wait_flush_ratio,
           py::call_guard<py::gil_scoped_release>())
      .def("set_saving_thread_number",
           &synced_tensor_dict::set_saving_thread_number,
           py::call_guard<py::gil_scoped_release>())
      .def("set_fetch_thread_number",
           &synced_tensor_dict::set_fetch_thread_number,
           py::call_guard<py::gil_scoped_release>())
      .def("__setitem__", &synced_tensor_dict::emplace,
           py::call_guard<py::gil_scoped_release>())
      .def("__len__", &synced_tensor_dict::size,
           py::call_guard<py::gil_scoped_release>())
      .def("__contains__", &synced_tensor_dict::contains,
           py::call_guard<py::gil_scoped_release>())
      .def("__getitem__", &synced_tensor_dict::get,
           py::call_guard<py::gil_scoped_release>())
      .def("__delitem__", &synced_tensor_dict::erase,
           py::call_guard<py::gil_scoped_release>())
      .def("keys", &synced_tensor_dict::keys,
           py::call_guard<py::gil_scoped_release>())
      .def("in_memory_keys", &synced_tensor_dict::in_memory_keys,
           py::call_guard<py::gil_scoped_release>())
      /* .def("__del__", &synced_tensor_dict::~synced_tensor_dict, */
      /*      py::call_guard<py::gil_scoped_release>()) */
      .def("clear", &synced_tensor_dict::clear,
           py::call_guard<py::gil_scoped_release>())
      .def("__copy__",
           [](const synced_tensor_dict &) {
             throw std::runtime_error("copy is not supported");
           })
      .def(
          "__deepcopy__",
          [](const synced_tensor_dict &, py::dict) {
            return std::runtime_error("deepcopy is not supported");
          },
          py::arg("memo"))
      .def("flush", &synced_tensor_dict::flush,
           "flush all in-memory data to the disk", py::arg("wait") = true,
           py::call_guard<py::gil_scoped_release>());
}
