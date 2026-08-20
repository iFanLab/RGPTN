import numpy as np
import warnings
import random
import os
import torch


def setup(args):
    # close warning output
    warnings.filterwarnings("ignore")

    # output format
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False, profile='default')

    set_seed(args.seed)
    device = set_device(args.gpu)

    # dir
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    return {
        'device': device
    }


def set_seed(seed=0):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(gpu=-1):
    """Set GPU or CPU device to run code."""
    gpu = int(gpu)
    device = f'cuda:{gpu}' if torch.cuda.is_available() and gpu >= 0 else 'cpu'
    return device
