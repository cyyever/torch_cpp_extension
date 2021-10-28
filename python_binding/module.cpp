#include <pybind11/pybind11.h>
#include "stochastic_quantization.hpp"
#include "synced_tensor_dict.hpp"
namespace py = pybind11;

PYBIND11_MODULE(cyy_naive_cpp_extension, m) {
  define_torch_data_structure_extension(m);
  define_torch_quantization_extension(m);
}
