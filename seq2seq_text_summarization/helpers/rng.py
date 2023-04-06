import torch
import numpy as np
import random

# Not used if using default worker initialization
def seed_worker(worker_id: int):
    """Set seed of worker based on its `worker_id`.
    
    Used in `torch.utils.data.DataLoader` to set seed of workers for multiprocessing.
    Not to be called manually.

    Args:
        worker_id (int): The process worker's ID number, starting from 0 to `num_workers` in `torch.utils.data.DataLoader`.
    """

    if type(worker_id) != int:
        raise TypeError(f"expected input argument to be of type int, got type {type(worker_id)}.")

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generator_obj_seed(seed: int = None):
    """PyTorch Generator for PRNG.

    Helper function to create a `torch.Generator` with `manual_seed` set to `seed`, or return None if generator is not needed.

    Args:
        seed (int, optional): The seed to set the `torch.Generator`'s `manual_seed` to.
    """

    if seed is None:
        return None
    elif type(seed) == int:
        return torch.Generator().manual_seed(seed)
    else:
        raise TypeError(f"expected input argument to be of type int or None, got type {type(seed)}.")