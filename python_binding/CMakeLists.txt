file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
pybind11_add_module(cyy_torch_cpp_extension ${SOURCE})
target_link_directories(cyy_torch_cpp_extension PUBLIC ${pytorch_path}/lib)
target_link_libraries(cyy_torch_cpp_extension PUBLIC torch_python)
target_link_libraries(cyy_torch_cpp_extension PUBLIC CyyTorchExtensionLib)
target_link_libraries(cyy_torch_cpp_extension PUBLIC CyyNaiveLib::log)
target_link_libraries(cyy_torch_cpp_extension PUBLIC CyyNaiveLib::util)

target_compile_definitions(cyy_torch_cpp_extension
                           PUBLIC BUILD_TORCH_PYTHON_BINDING)
