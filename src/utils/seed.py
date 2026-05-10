"""Random seed utilities for reproducibility."""

import random
import os

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    
    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU)
    - PyTorch (CUDA)
    - CUDA deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic options (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    # Also set environment variable for PyTorch
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int):
    """Seed worker function for DataLoader workers.
    
    Use with DataLoader worker_init_fn:
        DataLoader(dataset, worker_init_fn=seed_worker)
    
    Args:
        worker_id: Worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
