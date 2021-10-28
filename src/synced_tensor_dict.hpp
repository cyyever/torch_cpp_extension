#pragma once
#include <filesystem>
#include <utility>

#include <torch/types.h>

#include <cyy/algorithm/dict/cache.hpp>

namespace cyy::pytorch {
class synced_tensor_dict : public ::cyy::algorithm::cache<torch::Tensor> {
public:
  explicit synced_tensor_dict(const std::string &storage_dir_);
  void set_storage_dir(std::string storage_dir_);
  std::string get_storage_dir() const;

private:
  std::vector<std::string> load_keys() override;
  void clear_data() override;
  torch::Tensor load_data_from_disk(const std::string &key) override;
  void save_data(const std::string &key, torch::Tensor data) override;
  void erase_data(const std::string &key) override;
  std::filesystem::path get_tensor_file_path(const std::string &key) const;

private:
  std::filesystem::path storage_dir;
};
} // namespace cyy::pytorch
