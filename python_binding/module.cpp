#include "synced_tensor_dict.hpp"
namespace py = pybind11;

PYBIND11_MODULE(cyy_torch_cpp_extension, m) {
  define_torch_data_structure_extension(m);
}
