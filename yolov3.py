from itertools import chain
from typing import Optional
from model_loader import load_model_from_file
from modules import YOLOv3
import lightning as pl
from sys import float_info
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import box_iou
from processing import (
    non_max_supression,
    process_anchor,
    normalize_model_output,
    update_map,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import metric_names


# YOLOv3 module with pre- and postprocessing
class YoloV3Module(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        input_size=416,
        anchors: Tensor | None = None,
        learning_rate=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.anchors = anchors or torch.tensor(
            [
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]],
            ],
            dtype=torch.float32,
        )
        self.head_names = ("x52", "x26", "x13")
        self.conf_threshold = 0.5
        self.iou_threshold = 0.5
        self.input_size = input_size
        self.size_limits = (2, input_size)

        self.model = YOLOv3(num_classes)
        if num_classes == 80:
            load_model_from_file(self.model, "pretrained_weights/yolov3.weights")
        else:
            load_model_from_file(
                self.model.backbone, "pretrained_weights/darknet53.conv.74"
            )
        for p in self.model.backbone.parameters():
            p.requires_grad = False

        self.validation_map: Optional[MeanAveragePrecision] = None
        self.test_map: Optional[MeanAveragePrecision] = None

    def forward(self, x):
        return self.model(x)

    # based on: https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff
    # based on: https://github.com/v-iashin/WebsiteYOLO/blob/master/darknet.py#L237
    def _process_annotations(
        self,
        outputs: torch.Tensor,
        annotations: torch.Tensor,
        anchors: torch.Tensor,
        ignore_thresh: float = 0.7,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output_size = outputs.shape
        batch_size = output_size[0]
        grid_size = output_size[2]
        anchor_scale = self.input_size / grid_size
        anchors = anchors / anchor_scale
        assert output_size[2] == output_size[3]
        obj_mask = torch.zeros(output_size[:-1], dtype=torch.bool, device=self.device)
        processed = torch.zeros(output_size, device=self.device)

        image_batch_ids = annotations[..., 0].long().clamp(0, batch_size - 1).t()
        class_ids = annotations[..., 1].long().clamp(0, self.num_classes - 1).t()
        processed_bbox = annotations[..., 2:] * grid_size

        xy, wh = processed_bbox[..., :2], processed_bbox[..., 2:]
        # clamp prevents some weird indices on CUDA (e.g. 174725728)
        col, row = xy.long().clamp(0, grid_size - 1).t()

        def _xywh_from_wh(t: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                (torch.zeros(t.shape, device=self.device, requires_grad=True), t),
                dim=-1,
            )

        def _x1y1x2y2_from_xywh(xywh: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                (xywh[..., :2] - xywh[..., 2:] / 2, xywh[..., :2] + xywh[..., 2:] / 2),
                -1,
            )

        wh_iou_per_anchor = torch.stack(
            [
                box_iou(
                    _x1y1x2y2_from_xywh(
                        _xywh_from_wh(anchor.repeat(len(annotations), 1))
                    ),
                    _x1y1x2y2_from_xywh(_xywh_from_wh(wh)),
                ).diag()
                for anchor in anchors
            ],
            -1,
        )
        best_anchors = wh_iou_per_anchor.max(-1).indices
        obj_mask[image_batch_ids, best_anchors, row, col] = 1
        noobj_mask = ~obj_mask
        for ious in wh_iou_per_anchor:
            noobj_mask[image_batch_ids, :, row, col][:, ious > ignore_thresh, ...] = 0

        processed[image_batch_ids, best_anchors, row, col, :2] = xy - xy.floor()
        processed[image_batch_ids, best_anchors, row, col, 2:4] = torch.log(
            wh / anchors[best_anchors] + float_info.epsilon
        )
        processed[..., 4] = obj_mask.float()
        processed[image_batch_ids, best_anchors, row, col, 5 + class_ids] = 1

        return processed, obj_mask, noobj_mask

    # based on: https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff
    def _compute_loss(
        self,
        predicted: torch.Tensor,
        expected: torch.Tensor,
        mask_obj: torch.Tensor,
        mask_noobj: torch.Tensor,
    ) -> torch.Tensor:
        loss_x, loss_y, loss_w, loss_h = (
            F.mse_loss(predicted[..., i][mask_obj], expected[..., i][mask_obj])
            for i in range(4)
        )
        # TODO: magic numbers - honestly I don't know what they mean yet
        obj_coeff, noobj_coeff = 1, 100
        loss_obj_obj = obj_coeff * F.binary_cross_entropy(
            predicted[..., 4][mask_obj].clamp(0, 1),
            expected[..., 4][mask_obj].clamp(0, 1),
        )
        loss_obj_noobj = noobj_coeff * F.binary_cross_entropy(
            predicted[..., 4][mask_noobj].clamp(0, 1),
            expected[..., 4][mask_noobj].clamp(0, 1),
        )
        loss_obj = loss_obj_obj + loss_obj_noobj
        loss_cls = F.binary_cross_entropy(
            predicted[..., 5:][mask_obj], expected[..., 5:][mask_obj]
        )
        return loss_x + loss_y + loss_w + loss_h + loss_obj + loss_cls

    def _common_step(self, batch: list, batch_idx: int):
        # input: transformed image
        # annotations: annotation_batch_size * (image_id, class_id, x [0..1], y [0..1], w [0..1], h [0..1])
        #     where (x, y): center, (w, h): size, [0..1] wrt width or height
        #     and image_id identifies images within a single image batch
        input, annotations, paths, raw_input = batch
        loss = {}
        # outputs: Size([batch_size, anchors * (bbox + obj + num_classes), grid_size, grid_size]) for each head
        #     where anchors: 3, bbox: 4, obj: 1, num_classes: 2
        outputs = self(input)
        outputs_copy = dict(
            zip(self.head_names, [out.clone().detach() for out in outputs])
        )
        for i, (head_name, head_outputs) in enumerate(zip(self.head_names, outputs)):
            num_anchors = self.anchors.size(0)
            bbox_attrs = 5 + self.num_classes
            # resizing to: Size([batch_size, anchors, grid_size, grid_size, (bbox + obj + num_classes)])
            #     and using sigmoid to ensure [0..1] for x, y, objectness and per-class probabilities
            head_outputs = normalize_model_output(head_outputs, num_anchors, bbox_attrs)
            if annotations is None or len(annotations) == 0:
                loss[head_name] = 0
                continue
            (
                processed_annotations,
                mask_obj,
                mask_noobj,
            ) = self._process_annotations(
                head_outputs, annotations, self.anchors[i].to(self.device)
            )
            loss[head_name] = self._compute_loss(
                head_outputs,
                processed_annotations,
                mask_obj,
                mask_noobj,
            )
        total_loss = (
            loss[self.head_names[0]]
            + loss[self.head_names[1]]
            + loss[self.head_names[2]]
        )
        if self.anchors.device != self.device:
            self.anchors = self.anchors.to(self.device)
        anchors = self.anchors
        input_size = self.input_size
        num_classes = self.num_classes
        # Turn raw model output into x, y, w, h, objectness confidence, class probabilities
        bx52 = process_anchor(outputs_copy["x52"], input_size, anchors[0], num_classes)
        bx26 = process_anchor(outputs_copy["x26"], input_size, anchors[1], num_classes)
        bx13 = process_anchor(outputs_copy["x13"], input_size, anchors[2], num_classes)
        bpreds = torch.cat([bx52, bx26, bx13], dim=1)
        results = non_max_supression(
            bpreds, self.conf_threshold, self.size_limits, self.iou_threshold
        )
        return total_loss, results, annotations, paths, raw_input

    def training_step(self, batch: list, batch_idx: int):
        loss, _, _, _, _ = self._common_step(batch, batch_idx)
        # with open("./total_loss_grad_graph.txt", "w") as f:
        #     f.write(torchviz.make_dot(total_loss).__str__())
        return {metric_names.loss: loss}

    def training_epoch_end(self, outs):
        avg_loss = torch.stack([out[metric_names.loss] for out in outs]).mean()
        self.log("train_" + metric_names.avg_loss, avg_loss)

    def on_validation_epoch_start(self):
        self.validation_map = MeanAveragePrecision()

    def validation_step(self, batch: list, batch_idx: int):
        with torch.no_grad():
            loss, results, annotations, _, _ = self._common_step(batch, batch_idx)
        update_map(self.validation_map, results, annotations)
        return {"val_" + metric_names.loss: loss}

    def validation_epoch_end(self, outs):
        prefix = "val_"
        avg_loss = torch.stack([out[prefix + metric_names.loss] for out in outs]).mean()
        all_maps = self.validation_map.compute()
        self.validation_map = None
        self.log(prefix + metric_names.map_50, all_maps["map_50"])
        self.log(prefix + metric_names.map_75, all_maps["map_75"])
        self.log(prefix + metric_names.map_50_95, all_maps["map"])
        self.log(prefix + metric_names.avg_loss, avg_loss)

    def on_test_epoch_start(self):
        self.test_map = MeanAveragePrecision()

    def test_step(self, batch: list, batch_idx: int):
        with torch.no_grad():
            loss, results, annotations, _, _ = self._common_step(batch, batch_idx)
        update_map(self.test_map, results, annotations)
        return {"test_" + metric_names.loss: loss}

    def test_epoch_end(self, outs):
        prefix = "test_"
        avg_loss = torch.stack([out[prefix + metric_names.loss] for out in outs]).mean()
        all_maps = self.test_map.compute()
        self.test_map = None
        self.log(prefix + metric_names.map_50, all_maps["map_50"])
        self.log(prefix + metric_names.map_75, all_maps["map_75"])
        self.log(prefix + metric_names.map_50_95, all_maps["map"])
        self.log(prefix + metric_names.avg_loss, avg_loss)

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            _, results, _, paths, raw_images = self._common_step(batch, batch_idx)
        return paths, results, raw_images

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0005
        )
        return optimizer
