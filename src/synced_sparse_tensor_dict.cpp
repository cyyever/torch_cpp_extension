#include "synced_sparse_tensor_dict.hpp"
#include <torch/csrc/api/include/torch/serialize.h>

#include <stdexcept>

#include "log/log.hpp"

namespace cyy::pytorch {

synced_sparse_tensor_dict::synced_sparse_tensor_dict(
    torch::Tensor mask_, torch::IntArrayRef tensor_shape_,
    std::filesystem::path storage_dir_)
    : synced_tensor_dict(storage_dir_), mask{std::move(mask_)} {
  if (!mask.is_sparse()) {
    mask = mask.to_sparse();
    LOG_WARN("change mask to sparse");
  }
  if (!mask.is_sparse()) {
    throw std::invalid_argument("need sparse mask");
  }

  tensor_shape = tensor_shape_.vec();
}
void synced_sparse_tensor_dict::emplace(const std::string &key,
                                        const torch::Tensor &value) {

  auto sparse_value = value.sparse_mask(mask)._values();
  synced_tensor_dict::emplace(key, sparse_value);
}
std::optional<torch::Tensor>
synced_sparse_tensor_dict::get(const std::string &key) {
  auto value = synced_tensor_dict::get(key);
  if (!value.has_value()) {
    return value;
  }
  return sparse_coo_tensor(mask._indices(), value.value(), tensor_shape)
      .to_dense();
}
} // namespace cyy::pytorch
