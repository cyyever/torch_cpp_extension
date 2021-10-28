file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cyy_naive_cpp_extension ${SOURCE})
target_link_directories(cyy_naive_cpp_extension PUBLIC ${pytorch_path}/lib)
target_include_directories(cyy_naive_cpp_extension
                           PUBLIC ${pytorch_path}/include)
target_link_libraries(cyy_naive_cpp_extension PUBLIC torch_python)
target_link_libraries(cyy_naive_cpp_extension PUBLIC CyyTorchExtensionLib)
target_compile_definitions(cyy_naive_cpp_extension
                           PUBLIC BUILD_TORCH_PYTHON_BINDING)
