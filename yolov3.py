from model_loader import load_model_from_file
from modules import YOLOv3
import lightning as pl
from sys import float_info
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import box_iou, nms

from processing import process_anchor, xywh_to_rect, normalize_model_output


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
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]],
            ],
            dtype=torch.float32,
        )
        self.cpu = torch.device("cpu")
        self.conf_threshold = 0.2
        self.iou_threshold = 0.5
        self.size_limits = (2, 416)

        self.model = YOLOv3(num_classes)
        load_model_from_file(self.model.backbone, "pretrained/darknet53.conv.74")
        # load_model_from_file(self.model, "pretrained/yolov3.weights")

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
        grid_size = output_size[2]
        assert output_size[2] == output_size[3]
        obj_mask = torch.zeros(output_size[:-1], dtype=torch.bool, device=self.cpu)
        class_mask = torch.zeros(output_size[:-1], device=self.cpu)
        iou_scores = torch.zeros(output_size[:-1], device=self.cpu)
        processed = torch.zeros(output_size, device=self.cpu)

        image_batch_ids, class_ids = annotations[..., :2].long().t()
        processed_bbox = annotations[..., 2:] * grid_size

        p_xy, p_wh = processed_bbox[..., :2], processed_bbox[..., 2:]
        p_i, p_j = p_xy.long().t()
        # preventing some weird indices on CUDA (e.g. 174725728)
        p_i[p_i < 0] = 0
        p_j[p_j < 0] = 0
        p_i[p_i > grid_size - 1] = grid_size - 1
        p_j[p_j > grid_size - 1] = grid_size - 1

        def _make_box(t: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                (torch.zeros(t.shape, device=self.cpu), t),
                dim=-1,
            )

        wh_iou_per_anchor = torch.stack(
            [
                box_iou(
                    _make_box(anchor.repeat(len(annotations), 1)), _make_box(p_wh)
                ).diag()
                for anchor in anchors
            ],
            -1,
        )
        best_anchors = wh_iou_per_anchor.max(-1).indices
        obj_mask[image_batch_ids, best_anchors, p_j, p_i] = 1
        noobj_mask = ~obj_mask
        for ious in wh_iou_per_anchor:
            noobj_mask[image_batch_ids, :, p_j, p_i][:, ious > ignore_thresh, ...] = 0

        # TODO: why p_j before p_i? inverting coordinates?
        processed[image_batch_ids, best_anchors, p_j, p_i, :2] = p_xy - p_xy.floor()
        processed[image_batch_ids, best_anchors, p_j, p_i, 2:4] = torch.log(
            p_wh / anchors[best_anchors] + float_info.epsilon
        )
        processed[..., 4] = obj_mask.float()
        processed[image_batch_ids, best_anchors, p_j, p_i, 5 + class_ids] = 1

        outputs_bbox = outputs[image_batch_ids, best_anchors, p_j, p_i, :4].cpu()
        outputs_probabilities = outputs[
            image_batch_ids, best_anchors, p_j, p_i, 5:
        ].cpu()
        outputs_class_ids = torch.max(outputs_probabilities, dim=-1).indices
        class_mask[image_batch_ids, best_anchors, p_j, p_i] = (
            outputs_class_ids == class_ids
        ).float()

        outputs_bbox[..., 2:] += outputs_bbox[..., :2]
        processed_bbox[..., 2:] += processed_bbox[..., :2]
        iou_scores[image_batch_ids, best_anchors, p_j, p_i] = box_iou(
            outputs_bbox,
            processed_bbox,
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
        # input: transformed image
        # annotations: annotation_batch_size * (image_id, class_id, x [0..1], y [0..1], w [0..1], h [0..1])
        #     where (x, y): center, (w, h): size, [0..1] wrt width or height
        #     and image_id identifies images within a single image batch
        input, annotations = batch
        annotations = annotations.cpu()
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
            ) = self._process_annotations(head_outputs, annotations, self.anchors[i])
            loss[head] = self._compute_loss(
                head_outputs.to(self.device),
                processed_annotations[head].to(self.device),
                mask_obj,
                mask_noobj,
            )
        total_loss = torch.tensor(
            tuple(loss.values()), device=self.device, requires_grad=True
        ).sum()
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
            self.parameters(), lr=self.learning_rate, weight_decay=10e-4
        )
        return optimizer
