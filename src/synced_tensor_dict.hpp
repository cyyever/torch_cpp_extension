#pragma once
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>

#include <torch/csrc/api/include/torch/serialize.h>

#include "dict/lru_cache.hpp"

namespace cyy::pytorch {

class synced_tensor_dict
    : public ::cyy::algorithm::lru_cache<std::string, torch::Tensor> {
public:
  explicit synced_tensor_dict(std::filesystem::path storage_dir_);
  ~synced_tensor_dict() override = default;
  auto get_item(const key_type &key) { return get(key); }
  std::string get_storage_dir() const;
};
} // namespace cyy::pytorch
