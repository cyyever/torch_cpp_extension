#pragma once

#include "synced_tensor_dict.hpp"

namespace cyy::pytorch {
class synced_sparse_tensor_dict : public synced_tensor_dict {
public:
  synced_sparse_tensor_dict(torch::Tensor mask_,
                            torch::IntArrayRef tensor_shape_,
                            std::filesystem::path storage_dir_);
  void emplace(const std::string &key, const torch::Tensor &value);
  std::optional<torch::Tensor> get(const std::string &key);

private:
  torch::Tensor mask;
  std::vector<torch::IntArrayRef::value_type> tensor_shape;
};
} // namespace cyy::pytorch
