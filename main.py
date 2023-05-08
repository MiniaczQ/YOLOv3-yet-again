from torchsummary import summary
from torch import no_grad
from modules import YOLOv3

from model_loader import load_model_from_file
from processing import get_img


def main():
    model = YOLOv3(2).cuda()
    load_model_from_file(model.backbone, "pretrained/darknet53.conv.74")

    img = get_img("testimgs/n01582220_magpie.jpg")
    with no_grad():
        res = model.forward(img)


if __name__ == "__main__":
    main()
