from torch import nn, no_grad, from_numpy
import numpy as np


from modules import Darknet53Conv


def load_model_from_file(model, file):
    with open(file, "rb") as f:
        np.fromfile(f, dtype=np.int32, count=3)
        images = np.fromfile(f, dtype=np.int64, count=1)[0]
        print(f"Model pre-learned with {images} images")
        weights = np.fromfile(f, dtype=np.float32)
    load_model(model, weights)


# Based on https://www.programcreek.com/python/?CodeExample=load+darknet+weights
def load_model(model, weights):
    with no_grad():
        total = load_module(model, weights)
    if total != len(weights):
        raise Exception(f"Weights mismatch, required {total}, provided {len(weights)}")


def load_module(module: nn.Module, weights):
    total = 0
    match module:
        case conv if isinstance(module, Darknet53Conv):
            total += load_darknet53_conv(conv, weights[total:])
        case conv2d if isinstance(module, nn.Conv2d):
            total += load_conv2d(conv2d, weights[total:])
        case batch_norm2d if isinstance(module, nn.BatchNorm2d):
            total += load_batch_norm2d(batch_norm2d, weights[total:])
        case module:
            for submodule in module.children():
                total += load_module(submodule, weights[total:])
    return total


def load_param(param, weights):
    print(f"shape: {param.shape}")
    size = param.numel()
    data = weights[:size]
    param.data.copy_(from_numpy(data).view_as(param))
    return size


def load_darknet53_conv(module: Darknet53Conv, weights):
    total = 0
    total += load_batch_norm2d(module.bn, weights[total:])
    total += load_conv2d(module.conv, weights[total:])
    return total


def load_conv2d(module: nn.Conv2d, weights):
    total = 0
    if module.bias is not None:  # Required when loading Darknet53 classifier
        total += load_param(module.bias, weights[total:])
    total += load_param(module.weight, weights[total:])
    print(f"loaded conv {total}")
    return total


def load_batch_norm2d(module: nn.BatchNorm2d, weights):
    total = 0
    total += load_param(module.bias, weights[total:])
    total += load_param(module.weight, weights[total:])
    total += load_param(module.running_mean, weights[total:])
    total += load_param(module.running_var, weights[total:])
    print(f"loaded bn {total}")
    return total
