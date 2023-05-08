from torchsummary import summary
from torch import no_grad
from modules import Darknet53, Darknet53Classifier, YOLOv3

from model_loader import load_model_from_file
from processing import get_img


def main():
    # file = "darknet53.conv.74"
    # model = Darknet53().cuda()
    file = "darknet53.weights"
    model = Darknet53Classifier().cuda()
    # model = YOLOv3(2).cuda()
    load_model_from_file(model, file)

    img = get_img("n01582220_magpie.jpg")
    with no_grad():
        res = model.forward(img)

    res = res.squeeze()
    _, indices = res.sort()
    print(indices[-10:])


if __name__ == "__main__":
    main()
