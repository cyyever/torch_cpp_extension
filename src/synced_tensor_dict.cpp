#include "synced_tensor_dict.hpp"
#include <torch/csrc/api/include/torch/serialize.h>

#include <stdexcept>

namespace cyy::pytorch {

class tensor_storage_backend
    : public ::cyy::algorithm::storage_backend<torch::Tensor> {
public:
  explicit tensor_storage_backend(std::filesystem::path storage_dir_)
      : storage_dir(storage_dir_) {

    if (!std::filesystem::exists(storage_dir)) {
      std::filesystem::create_directories(storage_dir);
    }
  }
  std::vector<std::string> load_keys() override {

    std::vector<std::string> keys;
    if (storage_dir.empty()) {
      return keys;
    }
    if (std::filesystem::exists(storage_dir)) {
      if (!std::filesystem::is_directory(storage_dir)) {
        throw std::invalid_argument(storage_dir.string() +
                                    " is not a directory");
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
  void clear_data() override {
    std::lock_guard lk(data_mutex);
    if (std::filesystem::exists(storage_dir)) {
      std::filesystem::remove_all(storage_dir);
      std::filesystem::create_directories(storage_dir);
    }
  }
  torch::Tensor load_data(const std::string &key) override {

    std::lock_guard lk(data_mutex);
    torch::Tensor value;
    torch::load(value, get_tensor_file_path(key).string());
    return value;
  }
  void save_data(const std::string &key, torch::Tensor value) override {

    std::lock_guard lk(data_mutex);
    auto path = get_tensor_file_path(key);
    std::filesystem::remove(path);
    torch::save(value, path.string());
  }
  void erase_data(const std::string &key) override {

    std::lock_guard lk(data_mutex);
    std::filesystem::remove(get_tensor_file_path(key).string());
  }
  std::filesystem::path get_tensor_file_path(const std::string &key) const;

public:
  std::filesystem::path storage_dir;

private:
  std::mutex data_mutex;
};

synced_tensor_dict::synced_tensor_dict(std::filesystem::path storage_dir_)
    : cyy::algorithm::cache<torch::Tensor>(
          std::make_unique<tensor_storage_backend>(storage_dir_)) {}
synced_tensor_dict::~synced_tensor_dict() {
  release();
}
/* void synced_tensor_dict::set_storage_dir(std::filesystem::path storage_dir) {
 */
/*   if (storage_dir.empty()) { */
/*     throw std::invalid_argument(storage_dir.string() + " is not a
 * directory"); */
/*   } */
/*   this->flush_all(true); */
/*   std::lock_guard lk(data_mutex); */
/*   if (!std::filesystem::exists(storage_dir)) { */
/*     std::filesystem::create_directories(storage_dir); */
/*   } else { */
/*     if (!std::filesystem::is_directory(storage_dir)) { */
/*       throw std::invalid_argument(storage_dir.string() + " is not a
 * directory"); */
/*     } */
/*   } */
/*   dynamic_cast<tensor_storage_backend &>(*backend).storage_dir = storage_dir;
 */
/* } */

std::string synced_tensor_dict::get_storage_dir() const {
  std::lock_guard lk(data_mutex);
  return dynamic_cast<tensor_storage_backend &>(*backend).storage_dir.string();
}

std::filesystem::path
tensor_storage_backend::get_tensor_file_path(const std::string &key) const {
  if (storage_dir.empty()) {
    throw std::runtime_error("storage_dir is empty");
  }
  return storage_dir / std::filesystem::path(key);
}

} // namespace cyy::pytorch
