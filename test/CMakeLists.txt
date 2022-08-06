find_package(doctest REQUIRED)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

file(GLOB test_sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp)
foreach(test_source IN LISTS test_sources)
  get_filename_component(test_prog ${test_source} NAME_WE)
  add_executable(${test_prog} ${test_source})
  target_link_libraries(${test_prog} PRIVATE doctest::doctest)
  target_link_libraries(${test_prog} PRIVATE Threads::Threads)
  target_link_libraries(${test_prog} PRIVATE CyyTorchExtensionLib)
  target_compile_definitions(${test_prog}
                             PRIVATE DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN)
  add_test_with_runtime_analysis(TARGET ${test_prog} MEMCHECK OFF)
  get_target_property(new_env ${test_prog} ENVIRONMENT)
  list(APPEND new_env "LD_LIBRARY_PATH=${pytorch_path}/lib")
  set_target_properties(${test_prog} PROPERTIES ENVIRONMENT "${new_env}")
endforeach()
