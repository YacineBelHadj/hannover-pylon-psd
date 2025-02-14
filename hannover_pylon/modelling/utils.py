from collections import defaultdict
import random
from typing import Any, Callable, Dict, KeysView, Tuple, TypeVar
from torch import Tensor
from torch.utils.data import DataLoader
import torch
import numpy as np
Self = TypeVar("Self")

class TensorBuffer(object):
    """
    Used to buffer tensors
    """

    def __init__(self, device="cpu"):
        """

        :param device: device used to store buffers. Default is *cpu*.
        """
        self._buffer: Dict[Any, Any] = defaultdict(list)
        self.device = device

    def is_empty(self) -> bool:
        """
        Returns true if this buffer does not hold any tensors.
        """
        return len(self._buffer) == 0

    def append(self: Self, key, value: Tensor) -> Self:
        """
        Appends a tensor to the buffer.

        :param key: tensor identifier
        :param value: tensor
        """
        if not isinstance(value, Tensor):
            raise ValueError(f"Can not handle value type {type(value)}")

        value = value.detach().to(self.device)
        self._buffer[key].append(value)
        return self

    def __contains__(self, elem) -> bool:
        return elem in self._buffer

    def __getitem__(self, item) -> Tensor:
        return self.get(item)

    def sample(self, key) -> Tensor:
        """
        Samples a random tensor from the buffer

        :param key: tensor identifier
        :return: random tensor
        """
        index = torch.randint(0, len(self._buffer[key]), size=(1,))
        return self._buffer[key][index]

    def keys(self) -> KeysView:
        return self._buffer.keys()

    def get(self, key) -> Tensor:
        """
        Retrieves tensor from the buffer

        :param key: tensor identifier
        :return: concatenated tensor
        """
        if key not in self._buffer:
            raise KeyError(key)

        v = torch.cat(self._buffer[key])
        return v


    def save(self: Self, path) -> Self:
        """
        Save buffer to disk

        :return: self
        """
        d = {k: self.get(k).cpu() for k in self._buffer.keys()}
        torch.save(d, path)
        return self


def apply_reduction(tensor: Tensor, reduction: str) -> Tensor:
    """
    Apply specific reduction to a tensor
    """
    if reduction == "mean":
        return tensor.mean()
    elif reduction == "sum":
        return tensor.sum()
    elif reduction is None or reduction == "none":
        return tensor
    else:
        raise ValueError


def fix_random_seed(seed: int = 12345) -> None:
    """
    Set all random seeds.

    :param seed: seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)





def extract_features(
    data_loader: DataLoader, model: Callable[[Tensor], Tensor], device: str
) -> Tuple[Tensor, Tensor]:
    """
    Helper to extract outputs from model. Ignores OOD inputs.

    :param data_loader: dataset to extract from
    :param model: neural network to pass inputs to
    :param device: device used for calculations
    :return: Tuple with outputs and labels
    """
    # TODO: add option to buffer to GPU
    embedding = []
    label = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
        
            z = model(x)
            z = z.view(z.size(0), -1)
            embedding.append(z)
            label.append(y)

    return embedding, label