import random

import numpy as np
import torch

# Random seed for reproducibility
SEED = 42


def set_seed(seed=SEED):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set seed on import
set_seed()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
LEARNING_RATE = 1e-5
NUM_EPOCHS = 7
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PATCH_SIZE = 512
STRIDE = 256
