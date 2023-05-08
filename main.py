import numpy as np
from torchsummary import summary
from modules import Darknet53, Darknet53Classifier

from model_loader import load_model_from_file


def main():
    # file = "darknet53.conv.74"
    # model = Darknet53().cuda()
    file = "darknet53.weights"
    model = Darknet53Classifier().cuda()
    load_model_from_file(model, file)

    return
    from PIL import Image
    from torchvision.transforms import ToTensor, Normalize, Compose

    transformation = Compose(
        [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    t = (
        transformation(Image.open("n01582220_magpie.jpg").resize((418, 418)))
        .cuda()
        .unsqueeze(0)
    )
    res = model.forward(t)
    print(res.shape)
    res = res.squeeze()
    print(res.argmax())


if __name__ == "__main__":
    main()
