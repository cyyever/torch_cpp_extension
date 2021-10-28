cmake_minimum_required(VERSION 3.20)

project(CyyTorchExtensionLib LANGUAGES CXX)

include(cmake/all.cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if(Python3_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch ; print(torch.__path__[0])"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _path)
  if(NOT ${_exitcode} EQUAL 0)
    message(FATAL_ERROR "failed to found pytorch path")
  endif()
  string(STRIP ${_path} _path)
  set(pytorch_path
      ${_path}
      CACHE STRING "pytorch path" FORCE)
  list(PREPEND CMAKE_PREFIX_PATH ${pytorch_path}/share/cmake)
endif()

find_package(Torch REQUIRED)

file(GLOB_RECURSE SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

set(HAS_CUDA OFF)
if(TARGET torch_cuda)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  file(GLOB CUDA_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
  list(APPEND SOURCE ${CUDA_SOURCE})
  set(HAS_CUDA ON)
endif()

find_package(Boost REQUIRED)
find_package(CyyNaiveLib REQUIRED)
find_package(spdlog REQUIRED)

add_library(CyyTorchExtensionLib ${SOURCE})
target_compile_definitions(CyyTorchExtensionLib PRIVATE HAS_CUDA=${HAS_CUDA})
# target_link_libraries(CyyTorchExtensionLib PRIVATE CyyAlgorithmLib)
target_link_libraries(CyyTorchExtensionLib PRIVATE CyyNaiveLib::log)
target_compile_definitions(CyyTorchExtensionLib
                           PUBLIC _HAS_FEATURES_REMOVED_IN_CXX20=1)
target_link_libraries(CyyTorchExtensionLib PRIVATE CyyNaiveLib::util)
target_link_libraries(CyyTorchExtensionLib PRIVATE CyyNaiveLib::hardware)
target_link_directories(CyyTorchExtensionLib PUBLIC ${pytorch_path}/lib)
target_link_libraries(CyyTorchExtensionLib PUBLIC ${TORCH_LIBRARIES})
if(TARGET torch_cuda)
  target_link_libraries(CyyTorchExtensionLib PUBLIC torch_cuda)
  target_compile_definitions(
    CyyTorchExtensionLib PRIVATE __CUDA_NO_HALF_OPERATORS__ CUDA_HAS_FP16=1
                                 __CUDA_NO_HALF_CONVERSIONS__)
endif()

target_include_directories(
  CyyTorchExtensionLib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                              $<INSTALL_INTERFACE:include/cyy/torch>)

add_subdirectory(python_binding)
# test
add_subdirectory(test)

option(BUILD_FUZZING "Build fuzzing" OFF)
if(BUILD_FUZZING)
  add_subdirectory(fuzz_test)
endif()
# add_subdirectory(profiling)

# install lib
install(
  TARGETS CyyTorchExtensionLib
  EXPORT CyyTorchExtensionLibConfig
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib COMPONENT library
  INCLUDES
  DESTINATION include)

install(
  DIRECTORY src/
  DESTINATION include/cyy/torch
  FILES_MATCHING
  PATTERN "*.hpp")

# install cmake config file
install(EXPORT CyyTorchExtensionLibConfig
        DESTINATION lib/cmake/CyyTorchExtensionLib)
