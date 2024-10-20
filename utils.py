import torch
from typing import Optional, Tuple, Union
import numpy.typing as npt
import torch.nn as nn
import numpy as np
from cebra.data import SingleSessionDataset
import warnings

# Function to calculate the memory required by the model parameters
def get_model_memory_usage(model: nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 ** 2)  # Convert to MB (assuming float32)

# Function to calculate the memory required by the activations for a single batch
def get_activation_memory_usage(model: nn.Module, input_size: Tuple[int, ...], device) -> float:
    dummy_input = torch.randn(1,*input_size).to(device)
    activations = model(dummy_input)
    return activations.element_size() * activations.nelement() / (1024 ** 2)  # Convert to MB

# Function to estimate the optimal batch size
def estimate_optimal_batch_size(model: nn.Module, input_size: Tuple[int, ...], available_memory: float, device) -> int:
    model_memory = get_model_memory_usage(model)
    activation_memory_per_sample = get_activation_memory_usage(model, input_size, device)
    total_memory_per_sample = model_memory + activation_memory_per_sample
    optimal_batch_size = available_memory // total_memory_per_sample
    return int(optimal_batch_size)

def get_available_gpu_memory() -> Optional[float]:
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = gpu_memory - reserved_memory - allocated_memory
        return free_memory / (1024 ** 2)  # Convert to MB
    else:
        return None

class TensorDataset(SingleSessionDataset):
    """Discrete and/or continuously indexed dataset based on torch/numpy arrays.

    If dealing with datasets sufficiently small to fit :py:func:`numpy.array` or :py:class:`torch.Tensor`, this
    dataset is sufficient---the sampling auxiliary variable should be specified with a dataloader.
    Based on whether `continuous` and/or `discrete` auxiliary variables are provided, this class
    can be used with the discrete, continuous and/or mixed data loader classes.

    Args:
        neural:
            Array of dtype ``float`` or float Tensor of shape ``(N, D)``, containing neural activity over time.
        continuous:
            Array of dtype ```float`` or float Tensor of shape ``(N, d)``, containing the continuous behavior
            variables over the same time dimension.
        discrete:
            Array of dtype ```int64`` or integer Tensor of shape ``(N, d)``, containing the discrete behavior
            variables over the same time dimension.

    Example:

        >>> import cebra.data
        >>> import torch
        >>> data = torch.randn((100, 30))
        >>> index1 = torch.randn((100, 2))
        >>> index2 = torch.randint(0,5,(100, ))
        >>> dataset = cebra.data.datasets.TensorDataset(data, continuous=index1, discrete=index2)

    """

    def __init__(self,
                 neural: Union[torch.Tensor, npt.NDArray],
                 continuous: Union[torch.Tensor, npt.NDArray] = None,
                 discrete: Union[torch.Tensor, npt.NDArray] = None,
                 offset: int = 1,
                 device: str = "cpu"):
        super().__init__(device=device)
        self.neural = self._to_tensor(neural, 'float').float()
        self.continuous = self._to_tensor(continuous, 'float')
        self.discrete = self._to_tensor(discrete, 'long')
        if self.continuous is None and self.discrete is None:
            warnings.warn(
                "You should pass at least one of the arguments 'continuous' or 'discrete'."
            )
        self.offset = offset

    def _to_tensor(self, array, dtype=None):
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            if dtype == 'float':
                array = torch.from_numpy(array).float()
            elif dtype == 'long':
                array = torch.from_numpy(array).long()
            else:
                array = torch.from_numpy(array)
        return array

    @property
    def input_dimension(self) -> int:
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        if self.continuous is None:
            raise NotImplementedError()
        return self.continuous

    @property
    def discrete_index(self):
        if self.discrete is None:
            raise NotImplementedError()
        return self.discrete

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, index):
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)