from model_loader import load_model_from_file
from modules import YOLOv3
import lightning as pl
from sys import float_info
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
from processing import process_anchor, xywh_to_rect, normalize_model_output

# import torchviz


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
        self.conf_threshold = 0.2
        self.iou_threshold = 0.5
        self.input_size = input_size
        self.size_limits = (2, input_size)

        self.model = YOLOv3(num_classes)
        if num_classes == 80:
            load_model_from_file(self.model, "pretrained/yolov3.weights")
        else:
            load_model_from_file(self.model.backbone, "pretrained/darknet53.conv.74")

        self.epoch_train_loss_sum = 0

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
        class_mask = torch.zeros(output_size[:-1], device=self.device)
        iou_scores = torch.zeros(output_size[:-1], device=self.device)
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

        outputs_bbox = outputs[image_batch_ids, best_anchors, row, col, :4]
        outputs_probabilities = outputs[image_batch_ids, best_anchors, row, col, 5:]
        outputs_class_ids = torch.max(outputs_probabilities, dim=-1).indices
        class_mask[image_batch_ids, best_anchors, row, col] = (
            outputs_class_ids == class_ids
        ).float()

        iou_scores[image_batch_ids, best_anchors, row, col] = box_iou(
            _x1y1x2y2_from_xywh(outputs_bbox),
            _x1y1x2y2_from_xywh(processed_bbox),
        ).diag()

        return (processed, obj_mask, noobj_mask, class_mask, iou_scores)

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
        scale_loss_obj = [1, 100]  # magic numbers for obj and noobj
        loss_obj = torch.tensor(
            [
                scale
                * F.binary_cross_entropy(
                    predicted[..., 4][mask], expected[..., 4][mask]
                )
                for mask, scale in zip((mask_obj, mask_noobj), scale_loss_obj)
            ],
            device=self.device,
        ).sum()
        loss_cls = F.binary_cross_entropy(predicted[..., 5:], expected[..., 5:])
        return torch.tensor(
            [loss_x, loss_y, loss_w, loss_h, loss_obj, loss_cls],
            device=self.device,
        ).sum()

    def training_step(self, batch: list, batch_idx):
        if batch_idx == 0:
            self.epoch_train_loss_sum = 0
        # input: transformed image
        # annotations: annotation_batch_size * (image_id, class_id, x [0..1], y [0..1], w [0..1], h [0..1])
        #     where (x, y): center, (w, h): size, [0..1] wrt width or height
        #     and image_id identifies images within a single image batch
        input, annotations = batch
        loss, processed_annotations = {}, {}
        heads = ("x52", "x26", "x13")
        # outputs: Size([batch_size, anchors * (bbox + obj + num_classes), grid_size, grid_size]) for each head
        #     where anchors: 3, bbox: 4, obj: 1, num_classes: 2
        outputs = dict(zip(heads, self(input)))
        for i, (head, head_outputs) in enumerate(outputs.items()):
            num_anchors = self.anchors.size(0)
            bbox_attrs = 5 + self.num_classes
            # resizing to: Size([batch_size, anchors, grid_size, grid_size, (bbox + obj + num_classes)])
            #     and using sigmoid to ensure [0..1] for x, y, objectness and per-class probabilities
            head_outputs = normalize_model_output(head_outputs, num_anchors, bbox_attrs)
            (
                processed_annotations[head],
                mask_obj,
                mask_noobj,
                mask_class,
                iou_scores,
            ) = self._process_annotations(
                head_outputs, annotations, self.anchors[i].to(self.device)
            )
            loss[head] = self._compute_loss(
                head_outputs,
                processed_annotations[head],
                mask_obj,
                mask_noobj,
            )
        total_loss = torch.tensor(
            tuple(loss.values()), device=self.device, requires_grad=True
        ).sum()
        self.epoch_train_loss_sum += total_loss.item()
        self.log("batch_idx", batch_idx, prog_bar=True)
        self.log(
            "avg_epoch_train_loss",
            self.epoch_train_loss_sum / (batch_idx + 1),
            prog_bar=True,
        )
        return {"loss": total_loss}

    def predict_step(self, batch, batch_idx):
        (bx52, bx26, bx13) = self(batch)
        if self.anchors.device != self.device:
            self.anchors = self.anchors.to(self.device)
        anchors = self.anchors
        input_size = self.input_size
        num_classes = self.num_classes
        # Turn raw model output into x, y, w, h, objectness confidence, class probabilities
        bx52 = process_anchor(bx52, input_size, anchors[0], num_classes)
        bx26 = process_anchor(bx26, input_size, anchors[1], num_classes)
        bx13 = process_anchor(bx13, input_size, anchors[2], num_classes)
        bpreds = torch.cat([bx52, bx26, bx13], dim=1)
        results = []
        # Run postprocessing and non max suppression on every image's predictions
        for preds in bpreds:
            # Filter away low objectivness predictions
            preds = preds[preds[:, 4] > self.conf_threshold]
            # Filter away invalid prediction sizes
            min_mask = preds[:, [2, 3]] > self.size_limits[0]
            max_mask = preds[:, [2, 3]] < self.size_limits[1]
            preds = preds[(min_mask & max_mask).all(1)]
            # Early return
            if preds.size(0) == 0:
                results.append(preds)
                continue
            # Convert to rectangles
            boxes = xywh_to_rect(preds[:, [0, 1, 2, 3]])
            # Best prediction for each box
            confs, ids = preds[:, 5:].max(1)
            # Include objectness in class probability
            confs *= preds[:, 4]
            # Reduce predictions to x, y, w, h, class probability, class id
            preds = torch.cat([boxes, confs.view(-1, 1), ids.view(-1, 1)], 1)
            # Filter away low probability classes
            preds = preds[confs > self.conf_threshold]
            # Early return
            num_boxes = preds.size(0)
            if num_boxes == 0:
                results.append(preds)
                continue
            # Batched nms
            classes = preds[:, 5]  # classes
            boxes = (
                preds[:, :4].clone() + classes.view(-1, 1) * self.size_limits[1]
            )  # boxes (offset by class)
            scores = preds[:, 4]
            idxs = nms(boxes, scores, self.iou_threshold)
            if 1 < num_boxes:  # Merge NMS (boxes merged using weighted mean)
                try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = (
                        boxes.box_iou(boxes[idxs], boxes) > self.iou_threshold
                    )  # iou matrix
                    weights = iou * scores[None]  # box weights
                    preds[idxs, :4] = torch.mm(
                        weights, preds[:, :4]
                    ).float() / weights.sum(
                        1, keepdim=True
                    )  # merged boxes
                except:
                    pass
            results.append(preds[idxs])

        return results

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0005, momentum=0.9
        )
        return optimizer
