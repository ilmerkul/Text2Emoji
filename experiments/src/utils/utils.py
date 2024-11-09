import sys
import warnings

from loguru import logger
import numpy as np
import torch


def seed_all(seed: int) -> None:
    """
    Set seed for numpy and torch.
    :param seed: random seed
    :return: None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_model_size(model: torch.nn.Module) -> None:
    """
    Print params and size of model.
    :param model: torch model
    :return: None
    """
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"model params: {param_count}")
    print("model size: {:.3f}MB".format(size_all_mb))


def print_model(model: torch.nn.Module) -> None:
    """
    Print tensor of model's parameters
    :param model: torch model
    :return: None
    """
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print_model_size(model)


def set_logger() -> None:
    """
    Set loguru logger
    :return: None
    """
    # set up logging
    logger.remove()
    logger.add(sys.stdout,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    warnings.filterwarnings("ignore")


def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Load state and set state in torch model.
    :param model: torch model
    :param path: path for load model's state
    :return: torch model with state
    """
    state = torch.load(path)
    model.load_state_dict(state)

    return model
