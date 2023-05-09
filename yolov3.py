from model_loader import load_model_from_file
from modules import YOLOv3
import lightning as pl
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from processing import process_into_aabbs, process_with_nms, process_prediction


class YoloV3Module(pl.LightningModule):
    input_size = 416

    def __init__(
        self, num_classes=2, anchors: Tensor | None = None, learning_rate=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.anchors = anchors or torch.tensor(
            [
                [10, 13],
                [16, 30],
                [33, 23],
                [30, 61],
                [62, 45],
                [59, 119],
                [116, 90],
                [156, 198],
                [373, 326],
            ],
            dtype=torch.float32,
        )

        self.model = YOLOv3(num_classes)
        load_model_from_file(self.model.backbone, "pretrained/darknet53.conv.74")

    def forward(self, x):
        return self.model(x)

    # based on: https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff
    def _process_annotations(
        self, annotations: torch.Tensor, output_size: torch.Size
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_size = output_size[1]
        assert output_size[1] == output_size[2]
        obj_mask = torch.zeros(output_size[:-1], dtype=torch.bool, device=self.device)
        class_mask = torch.zeros(output_size[:-1], device=self.device)
        iou_scores = torch.zeros(output_size[:-1], device=self.device)

        processed_bbox = annotations[1:] * grid_size
        processed_xy = processed_bbox[:2]

        # TODO: WIP

        noobj_mask = ~obj_mask
        processed = torch.zeros(output_size, device=self.device)
        return (processed, obj_mask, noobj_mask, class_mask, iou_scores)

    # based on: https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff
    def _compute_loss(
        self, predicted: torch.Tensor, expected: torch.Tensor
    ) -> torch.Tensor:
        mask_obj = torch.ones_like(
            predicted[..., 0], dtype=torch.bool, device=self.device
        )
        loss_x, loss_y, loss_w, loss_h = (
            F.mse_loss(predicted[..., i][mask_obj], expected[..., i][mask_obj])
            for i in range(4)
        )
        # magic numbers - honestly I don't know what they mean yet
        scale_loss_obj = [1, 100]  # magic numbers for obj and noobj
        loss_obj = torch.tensor(
            [
                scale
                * F.binary_cross_entropy(
                    predicted[..., 4][mask], expected[..., 4][mask]
                )
                for mask, scale in zip((mask_obj, ~mask_obj), scale_loss_obj)
            ],
            device=self.device,
        ).sum()
        loss_cls = F.binary_cross_entropy(predicted[..., 5:], expected[..., 5:])
        return torch.tensor(
            [loss_x, loss_y, loss_w, loss_h, loss_obj, loss_cls], device=self.device
        ).sum()

    def training_step(self, batch: torch.Tensor, batch_idx):
        # input: transformed image
        # annotations: annotations (class_id, x [0..1], y [0..1], w [0..1], h [0..1])
        #     where (x, y): center, (w, h): size, [0..1] wrt width or height
        input, annotations = batch
        loss, processed_annotations = {}, {}
        heads = ("x52", "x26", "x13")
        # outputs: Size([3, grid_size, grid_size, (4 + 1 + num_classes)]) for each head
        outputs = dict(zip(heads, self(input)))
        for head, head_outputs in outputs.items():
            # using sigmoid to ensure [0..1] for x, y, objectness and per-class probabilities
            trunc_idx = [0, 1, 4] + list(range(5, 5 + self.num_classes))
            head_outputs[:, :, :, trunc_idx] = torch.sigmoid(
                head_outputs[:, :, :, trunc_idx]
            )
            processed_annotations[head] = self._process_annotations(
                annotations, head_outputs.shape
            )
            loss[head] = self._compute_loss(head_outputs, processed_annotations[head])
        return loss, outputs, processed_annotations

    def validation_step(self, batch, batch_idx):
        pass  # TODO

    def test_step(self, batch, batch_idx):
        pass  # TODO

    def predict_step(self, batch, batch_idx):
        bpreds = self(batch)
        baabbs = process_into_aabbs(
            bpreds,
            self.input_size,
            self.anchors,
            self.num_classes,
            0.7,
            0.2,
            self.device,
        )
        for aabbs in baabbs:
            print(aabbs)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, weight_decay=10e-4
        )
        return optimizer
