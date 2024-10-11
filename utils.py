import torch
from typing import Optional, Tuple
import torch.nn as nn

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