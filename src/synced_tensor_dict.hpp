#pragma once
#include <filesystem>
#include <mutex>
#include <utility>

#include <torch/serialize.h>

#include "dict/cache.hpp"

namespace cyy::pytorch {

class synced_tensor_dict : public ::cyy::algorithm::cache<torch::Tensor> {
public:
  explicit synced_tensor_dict(std::filesystem::path storage_dir_);
  ~synced_tensor_dict() override;
  void set_storage_dir(std::filesystem::path storage_dir);
  std::string get_storage_dir() const;

private:
  mutable std::recursive_mutex data_mutex;
};
} // namespace cyy::pytorch
