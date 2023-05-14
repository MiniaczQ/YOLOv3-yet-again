# Roughly based on https://www.programcreek.com/python/?CodeExample=load+darknet+weights
import numpy as np
from torch import nn, no_grad, from_numpy

from .modules import Darknet53Conv, YOLOv3


# Fill model with weights stored in a file
# 20 bytes of header then weights
def load_model_from_file(model, file):
    with open(file, "rb") as f:
        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)
    load_model(model, weights)


# Loads a model from numpy array
def load_model(model, weights):
    with no_grad():
        total = load_module(model, weights)
    if total != len(weights):
        raise Exception(f"Weights mismatch, required {total}, provided {len(weights)}")


# Loads weights of a single module (and its' submodules0)
def load_module(module: nn.Module, weights):
    total = 0
    match module:
        case conv if isinstance(module, Darknet53Conv):
            total += load_darknet53_conv(conv, weights[total:])
        case conv2d if isinstance(module, nn.Conv2d):
            total += load_conv2d(conv2d, weights[total:])
        case batch_norm2d if isinstance(module, nn.BatchNorm2d):
            total += load_batch_norm2d(batch_norm2d, weights[total:])
        case yolov3 if isinstance(yolov3, YOLOv3):
            total += load_yolov3(batch_norm2d, weights[total:])
        case module:
            for submodule in module.children():
                total += load_module(submodule, weights[total:])
    return total


# Loads a single parameter
def load_param(param, weights):
    size = param.numel()
    data = weights[:size]
    param.data.copy_(from_numpy(data).view_as(param))
    return size


# Loads a convolution layer parameters in correct order
def load_darknet53_conv(module: Darknet53Conv, weights):
    total = 0
    total += load_module(module.bn, weights[total:])
    total += load_module(module.conv, weights[total:])
    return total


# Loads a conv2d layer parameters in correct order
def load_conv2d(module: nn.Conv2d, weights):
    total = 0
    if module.bias is not None:  # Required when loading Darknet53 classifier
        total += load_param(module.bias, weights[total:])
    total += load_param(module.weight, weights[total:])
    return total


# Loads a batchnorm2d layer parameters in correct order
def load_batch_norm2d(module: nn.BatchNorm2d, weights):
    total = 0
    total += load_param(module.bias, weights[total:])
    total += load_param(module.weight, weights[total:])
    total += load_param(module.running_mean, weights[total:])
    total += load_param(module.running_var, weights[total:])
    return total


# Loads YOLOv3 modes in the correct order
def load_yolov3(module: YOLOv3, weights):
    total = 0
    total += load_module(module.backbone, weights[total:])
    total += load_module(module.neck.conv1, weights[total:])
    total += load_module(module.head1, weights[total:])
    total += load_module(module.neck.upsample1, weights[total:])
    total += load_module(module.neck.conv2, weights[total:])
    total += load_module(module.head2, weights[total:])
    total += load_module(module.neck.upsample2, weights[total:])
    total += load_module(module.neck.conv3, weights[total:])
    total += load_module(module.head3, weights[total:])
    return total
