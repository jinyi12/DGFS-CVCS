import os, sys
from itertools import count
import pathlib
import functools
import socket
import logging
from datetime import datetime

import math
import random
import scipy
import numpy as np
import torch


######### System Utils


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


######### Pytorch Utils

import random


def seed_torch(seed, verbose=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    if verbose:
        print("==> Set seed to {:}".format(seed))


def normal_logp(x, mean=0.0, sigma=1.0):
    """
    Calculate the log probability of a normal distribution.

    Args:
        x (torch.Tensor): The state, shape: [b, d]
        mean (float, optional): The mean of the normal distribution. Defaults to 0.0.
        sigma (float, optional): The standard deviation of the normal distribution. Defaults to 1.0.

    Returns:
        torch.Tensor: The log probability, shape: [b, 1]
    """
    assert x.ndim == 2
    if isinstance(sigma, torch.Tensor):
        assert sigma.ndim == 2
        log_sigma = torch.log(sigma)
    else:
        log_sigma = np.log(sigma)

    neg_logp = 0.5 * np.log(2 * np.pi) + log_sigma + 0.5 * (x - mean) ** 2 / (sigma**2)
    return torch.sum(-neg_logp, dim=-1, keepdim=False)  # Return shape [b]


def loss2ess_info(loss):
    # ESS = (\sum w_i)^2 / \sum w_i^2
    # return ESS / N <= 1
    log_weight = -loss + loss.mean()
    log_numerator = 2 * torch.logsumexp(log_weight, dim=0)
    log_denominator = torch.logsumexp(2 * log_weight, dim=0)
    ess = torch.exp(log_numerator - log_denominator) / len(log_weight)
    return {"ess": ess.item()}


def setup_logging(save_dir):
    """
    Sets up logging configuration to output to both console and file.

    Args:
        save_dir (str): Directory where the log file will be saved

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("gflownet")
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
