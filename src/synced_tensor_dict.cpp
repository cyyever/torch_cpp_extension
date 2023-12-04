#include "synced_tensor_dict.hpp"

#include <shared_mutex>
#include <stdexcept>

namespace cyy::pytorch {

class tensor_storage_backend
    : public ::cyy::algorithm::storage_backend<std::string, torch::Tensor> {
public:
  explicit tensor_storage_backend(std::filesystem::path storage_dir_)
      : storage_dir(storage_dir_) {

    if (!std::filesystem::exists(storage_dir)) {
      std::filesystem::create_directories(storage_dir);
      empty_dir = true;
    } else {
      if (!std::filesystem::is_directory(storage_dir)) {
        throw std::invalid_argument(storage_dir.string() +
                                    " is not a directory");
      }
    }
  }
  std::vector<std::string> get_keys() override {
    std::vector<std::string> keys;
    std::shared_lock lk(data_mutex);
    if (storage_dir.empty()) {
      return keys;
    }
    for (const auto &f : std::filesystem::directory_iterator(storage_dir)) {
      if (f.is_regular_file()) {
        auto key = f.path().filename().string();
        keys.push_back(std::move(key));
      }
    }
    return keys;
  }
  bool contains(const key_type &key) override {
    auto tensor_path = get_tensor_file_path(key);
    std::shared_lock lk(data_mutex);
    return std::filesystem::is_regular_file(tensor_path);
  }
  void clear() override {
    std::lock_guard lk(data_mutex);
    std::filesystem::remove_all(storage_dir);
    std::filesystem::create_directories(storage_dir);
  }
  std::optional<torch::Tensor> load_data(const std::string &key) override {
    auto tensor_path = get_tensor_file_path(key);
    std::lock_guard lk(data_mutex);
    if (std::filesystem::is_regular_file(tensor_path)) {
      torch::Tensor value;
      torch::load(value, tensor_path.string());
      return value;
    }
    return {};
  }

  bool save_data(const std::string &key, torch::Tensor value) override {
    std::lock_guard lk(data_mutex);
    auto path = get_tensor_file_path(key);
    std::filesystem::remove(path);
    torch::save(value, path.string());
    return true;
  }
  void erase_data(const std::string &key) override {
    std::lock_guard lk(data_mutex);
    std::filesystem::remove(get_tensor_file_path(key).string());
  }

private:
  std::filesystem::path get_tensor_file_path(const std::string &key) const {
    if (storage_dir.empty()) {
      throw std::runtime_error("storage_dir is empty");
    }
    return storage_dir / std::filesystem::path(key);
  }

public:
  std::filesystem::path storage_dir;
  bool empty_dir{false};

private:
  static inline std::shared_mutex data_mutex;
};

synced_tensor_dict::synced_tensor_dict(std::filesystem::path storage_dir_)
    : cyy::algorithm::lru_cache<std::string, torch::Tensor>(
          std::make_unique<tensor_storage_backend>(storage_dir_),
          dynamic_cast<tensor_storage_backend &>(*backend).empty_dir) {}

std::string synced_tensor_dict::get_storage_dir() const {
  return dynamic_cast<tensor_storage_backend &>(*backend).storage_dir.string();
}

} // namespace cyy::pytorch
