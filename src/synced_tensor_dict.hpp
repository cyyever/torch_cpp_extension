#pragma once
#include <filesystem>
#include <mutex>
#include <utility>

#include <torch/types.h>

#include <cyy/algorithm/dict/cache.hpp>

namespace cyy::pytorch {

class tensor_storage_backend
    : public ::cyy::algorithm::storage_backend<torch::Tensor> {
public:
  explicit tensor_storage_backend(const std::string &storage_dir_);
  std::vector<std::string> load_keys() override;
  void clear_data() override;
  torch::Tensor load_data(const std::string &key) override;
  void save_data(const std::string &key, torch::Tensor data) override;
  void erase_data(const std::string &key) override;
  std::filesystem::path get_tensor_file_path(const std::string &key) const;

public:
  std::filesystem::path storage_dir;
};

class synced_tensor_dict : public ::cyy::algorithm::cache<torch::Tensor> {
public:
  explicit synced_tensor_dict(const std::string &storage_dir_);
  void set_storage_dir(std::filesystem::path storage_dir_);
  std::filesystem::path get_storage_dir() const;

private:
  std::recursive_mutex data_mutex;
};
} // namespace cyy::pytorch
