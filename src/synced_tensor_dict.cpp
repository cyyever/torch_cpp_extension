#include "synced_tensor_dict.hpp"
#include <torch/csrc/api/include/torch/serialize.h>

#include <stdexcept>

namespace cyy::pytorch {

std::vector<std::string> synced_tensor_dict::load_keys() {
  std::vector<std::string> keys;
  if (std::filesystem::exists(storage_dir)) {
    if (!std::filesystem::is_directory(storage_dir)) {
      throw std::invalid_argument(storage_dir.string() + " is not a directory");
    }
    for (const auto &f : std::filesystem::directory_iterator(storage_dir)) {
      if (f.is_regular_file()) {
        auto key = f.path().filename().string();
        keys.push_back(std::move(key));
      }
    }
  } else {
    std::filesystem::create_directories(storage_dir);
  }
  return keys;
}

void synced_tensor_dict::clear_data() {
  if (!storage_dir.empty() && std::filesystem::exists(storage_dir)) {
    std::filesystem::remove_all(storage_dir);
    std::filesystem::create_directories(storage_dir);
  }
}
torch::Tensor synced_tensor_dict::load_data_from_disk(const std::string &key) {
  torch::Tensor value;
  torch::load(value, get_tensor_file_path(key).string());
  return value;
}
void synced_tensor_dict::save_data(const std::string &key, torch::Tensor data) {
  auto path = get_tensor_file_path(key);
  std::filesystem::remove(path);
  torch::save(data, path.string());
}
void synced_tensor_dict::erase_data(const std::string &key) {
  std::filesystem::remove(get_tensor_file_path(key).string());
}

void synced_tensor_dict::set_storage_dir(std::string storage_dir_) {
  if (storage_dir_.empty()) {
    throw std::invalid_argument(storage_dir_ + " is not a directory");
  }
  std::lock_guard lk(data_mutex);
  storage_dir = std::move(storage_dir_);
  if (!std::filesystem::exists(storage_dir)) {
    std::filesystem::create_directories(storage_dir);
  } else {
    if (!std::filesystem::is_directory(storage_dir)) {
      throw std::invalid_argument(storage_dir.string() + " is not a directory");
    }
  }
}

std::string synced_tensor_dict::get_storage_dir() const {
  std::lock_guard lk(data_mutex);
  return storage_dir.string();
}

std::filesystem::path
synced_tensor_dict::get_tensor_file_path(const std::string &key) const {
  std::lock_guard lk(data_mutex);
  if (storage_dir.empty()) {
    throw std::runtime_error("storage_dir is empty");
  }
  return storage_dir / std::filesystem::path(key);
}

} // namespace cyy::pytorch
