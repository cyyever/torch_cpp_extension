import os
import tempfile

import cyy_torch_cpp_extension
import torch


def test_synced_tensor_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert os.path.exists(tmpdir)
        tensor_dict = cyy_torch_cpp_extension.data_structure.SyncedTensorDictIMPL(
            tmpdir
        )
        tensor_dict.set_in_memory_number(10)

        # tensor_dict.set_permanent_storage()
        for i in range(100):
            tensor_dict[str(i)] = torch.Tensor([i])

        tensor_dict.prefetch([str(i) for i in range(100)])
        assert str(0) in tensor_dict
        for i in tensor_dict.keys():
            assert tensor_dict[i] == torch.Tensor([int(i)])
        tensor_dict.flush()
        del tensor_dict
