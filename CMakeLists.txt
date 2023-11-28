cmake_minimum_required(VERSION 3.20)

project(CyyTorchExtensionLib LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
set(BUILD_TESTING OFF)
set(SPDLOG_BUILD_SHARED OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog)
set(BUILD_OBJECT_LIB ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/naive_cpp_lib EXCLUDE_FROM_ALL)

file(GLOB_RECURSE SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(CyyTorchExtensionLib SHARED ${SOURCE})
target_link_libraries(CyyTorchExtensionLib PRIVATE CyyNaiveLib::log)
target_link_libraries(CyyTorchExtensionLib PRIVATE CyyNaiveLib::util)
target_compile_definitions(
  CyyTorchExtensionLib PUBLIC _HAS_FEATURES_REMOVED_IN_CXX20=1
                              _LIBCPP_ENABLE_CXX20_REMOVED_FEATURES)
get_filename_component(libtorch_dir torch_cpu DIRECTORY)
target_link_libraries(CyyTorchExtensionLib PUBLIC torch_cpu)

target_include_directories(
  CyyTorchExtensionLib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/algorithm/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/>
    $<BUILD_INTERFACE:${pytorch_path}/include/torch/csrc/api/include>
    $<BUILD_INTERFACE:${pytorch_path}/include>
    $<INSTALL_INTERFACE:include/cyy/torch>)

add_subdirectory(python_binding)
# test
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

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
