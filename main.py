from torch import nn, no_grad, from_numpy
import numpy as np

from torchsummary import summary

from modules import Darknet53, Darknet53Classifier

def load_param(param, weights):
    size = param.numel()
    shape = param.shape
    data = weights[:size]
    tensor = from_numpy(data).view(shape)
    param.copy_(tensor)
    return size

# Based on https://www.programcreek.com/python/?CodeExample=load+darknet+weights
def load_model(model, weights):
    total = 0
    with no_grad():
        for module in model.modules():
            match module:
                case conv if isinstance(module, nn.Conv2d):
                    total += load_param(conv.weight, weights[total:])
                    if conv.bias is not None: # Required when loading Darknet53 classifier
                        total += load_param(conv.bias, weights[total:])
                case bn if isinstance(module, nn.BatchNorm2d):
                    total += load_param(bn.bias, weights[total:])
                    total += load_param(bn.weight, weights[total:])
                    total += load_param(bn.running_mean, weights[total:])
                    total += load_param(bn.running_var, weights[total:])
    if total != len(weights):
        raise Exception(f'Weights mismatch, required {total}, provided {len(weights)}')

def main():
    weights_file = "darknet53.conv.74"
    model = Darknet53().cuda()
    #weights_file = "darknet53.weights"
    #model = Darknet53Classifier().cuda()
    summary(model, input_size=(3, 416, 416))
    with open(weights_file, "rb") as f:
        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)
    load_model(model, weights)
    


if __name__ == "__main__":
    main()


#40549216
#40620640

#41573216
#41645640