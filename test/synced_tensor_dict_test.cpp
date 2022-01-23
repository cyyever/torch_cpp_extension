/*!
 * \file container_test.cpp
 *
 * \brief 测试container相关函数
 * \author cyy
 */
#include <thread>
#include <filesystem>

#include <doctest/doctest.h>

#include "synced_tensor_dict.hpp"

TEST_CASE("synced_tensor_dict") {
  if(std::filesystem::exists("tensor_dir")) {
    std::filesystem::remove_all("tensor_dir");
  }
  cyy::pytorch::synced_tensor_dict dict("tensor_dir");

  CHECK_EQ(dict.size(), 0);

  dict.set_in_memory_number(3);

  // save sparse tensor
  auto sparse_tensor = torch::eye(3).to_sparse();
  dict.emplace("sparse_tensor", sparse_tensor);

  CHECK_EQ(dict.size(), 1);
  CHECK_EQ(dict.keys().size(), 1);
  dict.erase("sparse_tensor");

  CHECK_EQ(dict.size(), 0);

  std::vector<std::thread> thds;
  for (int i = 0; i < 10; i++) {
    thds.emplace_back([i, &dict]() {
      for (int j = 0; j < 10; j++) {
        dict.emplace(std::to_string(i * 10 + j), torch::rand({1, 200 * 1024}));
      }
    });
  }
  for (auto &thd : thds) {
    thd.join();
  }
  thds.clear();
  CHECK_EQ(dict.size(), 100);
  CHECK_EQ(dict.keys().size(), 100);

  std::vector<std::string> keys;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      keys.emplace_back(std::to_string(i * 10 + j));
    }
  }
  dict.flush_all(true);

  CHECK(dict.contains("0"));

  dict.prefetch(keys);

  CHECK(!dict.contains("100000"));
  CHECK(!dict.get("100000").has_value());
  dict.erase("10000");

  for (int i = 0; i < 10; i++) {
    thds.emplace_back([i, &dict]() {
      for (int j = 0; j < 10; j++) {
        auto tr = dict.get(std::to_string(i * 10 + j));
      }
    });
  }
  for (auto &thd : thds) {
    thd.join();
  }
  CHECK_EQ(dict.size(), 100);
  dict.enable_permanent_storage();
  dict.release();
  cyy::pytorch::synced_tensor_dict dict2("tensor_dir");
  CHECK_EQ(dict2.size(), 100);
  dict2.disable_permanent_storage();
  dict2.clear();
  CHECK_EQ(dict2.size(), 0);
}
