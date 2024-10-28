import numpy as np
import torch
from loguru import logger
import sys
import warnings


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_model_size(model):
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f'model params: {param_count}')
    print('model size: {:.3f}MB'.format(size_all_mb))


def print_model(model):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print_model_size(model)


def set_logger():
    # set up logging
    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    warnings.filterwarnings('ignore')


def load_model(model, path):
    state = torch.load(path)
    model.load_state_dict(state)

    return model
