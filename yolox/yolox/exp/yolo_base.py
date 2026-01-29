# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================================
""" utils """
import ast
import os
import sys
import time
import json
from datetime import datetime
import numpy as np
from mindspore.train.callback import Callback
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.transform import xyxy2xywh


class Exp:
    def __init__(self):
        super().__init__()
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


class YOLOXCB(Callback):
    """
    YOLOX Callback.
    """

    def __init__(self, logger, step_per_epoch, lr, is_modelart=False, per_print_times=1,
                 train_url=None):
        super(YOLOXCB, self).__init__()
        self.train_url = train_url
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.lr = lr
        self.is_modelarts = is_modelart
        self.step_per_epoch = step_per_epoch
        self.current_step = 0
        self.iter_time = time.time()
        self.epoch_start_time = time.time()
        self.average_loss = []
        self.logger = logger

    def epoch_begin(self, run_context):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_start_time = time.time()
        self.iter_time = time.time()

    def epoch_end(self, run_context):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            loss = "loss: %.4f, overflow: %s, scale: %s" % (float(loss[0].asnumpy()),
                                                            bool(loss[1].asnumpy()),
                                                            int(loss[2].asnumpy()))
        else:
            loss = "loss: %.4f" % float(loss.asnumpy())
        self.logger.info(
            "epoch: %s epoch time %.2fs %s" % (cur_epoch, time.time() - self.epoch_start_time, loss))

    def step_begin(self, run_context):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_end(self, run_context):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cur_epoch_step = (self.current_step + 1) % self.step_per_epoch
        if cur_epoch_step % self._per_print_times == 0 and cur_epoch_step != 0:
            cb_params = run_context.original_args()
            cur_epoch = cb_params.cur_epoch_num
            loss = cb_params.net_outputs
            if isinstance(loss, (tuple, list)):
                loss = "loss: %.4f, overflow: %s, scale: %s" % (float(loss[0].asnumpy()),
                                                                bool(loss[1].asnumpy()),
                                                                int(loss[2].asnumpy()))
            else:
                loss = "loss: %.4f" % float(loss.asnumpy())
            # self.logger.info("epoch: %s step: [%s/%s], %s, lr: %.6f, avg step time: %.2f ms" % (
            #     cur_epoch, cur_epoch_step, self.step_per_epoch, loss, self.lr[self.current_step],
            #     (time.time() - self.iter_time) * 1000 / self._per_print_times))
            self.logger.info("epoch: %s step: [%s/%s], %s, lr: %.6f, avg step time: %.2f ms" % (
                cur_epoch, cur_epoch_step, self.step_per_epoch, loss, self.lr,
                (time.time() - self.iter_time) * 1000 / self._per_print_times))
            self.iter_time = time.time()
        self.current_step += 1

    def end(self, run_context):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


class DetectionEngine:
    """ Detection engine """

    def __init__(self, config):
        self.config = config
        self.input_size = self.config.input_size
        self.strides = self.config.fpn_strides  # [8, 16, 32]

        self.expanded_strides = None
        self.grids = None

        self.num_classes = config.num_classes

        self.conf_thre = config.conf_thre
        self.nms_thre = config.nms_thre
        self.annFile = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
        self._coco = COCO(self.annFile)
        self._img_ids = list(sorted(self._coco.imgs.keys()))
        self.coco_catIds = self._coco.getCatIds()
        self.save_prefix = config.outputs_dir
        self.file_path = ''

        self.data_list = []

    def detection(self, outputs, img_shape, img_ids):
        # post process nms
        outputs = self.postprocess(outputs, self.num_classes, self.conf_thre, self.nms_thre)
        self.data_list.extend(self.convert_to_coco_format(outputs, info_imgs=img_shape, ids=img_ids))

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        """ nms """
        box_corner = np.zeros_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            if not image_pred.shape[0]:
                continue
            # Get score and class with highest confidence
            class_conf = np.max(image_pred[:, 5:5 + num_classes], axis=-1)  # (8400)
            class_pred = np.argmax(image_pred[:, 5:5 + num_classes], axis=-1)  # (8400)
            conf_mask = (image_pred[:, 4] * class_conf >= conf_thre).squeeze()  # (8400)
            class_conf = np.expand_dims(class_conf, axis=-1)  # (8400, 1)
            class_pred = np.expand_dims(class_pred, axis=-1).astype(np.float16)  # (8400, 1)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
            detections = detections[conf_mask]
            if not detections.shape[0]:
                continue
            if class_agnostic:
                nms_out_index = self._nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
            else:
                nms_out_index = self._batch_nms(detections[:, :4], detections[:, 4] * detections[:, 5],
                                                detections[:, 6], nms_thre)
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = np.concatenate((output[i], detections))
        return output

    def _nms(self, xyxys, scores, threshold):
        """Calculate NMS"""
        x1 = xyxys[:, 0]
        y1 = xyxys[:, 1]
        x2 = xyxys[:, 2]
        y2 = xyxys[:, 3]
        scores = scores
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h

            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)
            indexes = np.where(ovr <= threshold)[0]
            order = order[indexes + 1]
        return reserved_boxes

    def _batch_nms(self, xyxys, scores, idxs, threshold, use_offset=True):
        """Calculate Nms based on class info,Each index value correspond to a category,
        and NMS will not be applied between elements of different categories."""
        if use_offset:
            max_coordinate = xyxys.max()
            offsets = idxs * (max_coordinate + np.array([1]))
            boxes_for_nms = xyxys + offsets[:, None]
            keep = self._nms(boxes_for_nms, scores, threshold)
            return keep
        keep_mask = np.zeros_like(scores, dtype=np.bool_)
        for class_id in np.unique(idxs):
            curr_indices = np.where(idxs == class_id)[0]
            curr_keep_indices = self._nms(xyxys[curr_indices], scores[curr_indices], threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = np.where(keep_mask)[0]
        return keep_indices[np.argsort(-scores[keep_indices])]

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        """ convert to coco format """
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
                outputs, info_imgs[:, 0], info_imgs[:, 1], ids
        ):
            if output is None:
                continue
            bboxes = output[:, 0:4]
            scale = min(
                self.input_size[0] / float(img_h), self.input_size[1] / float(img_w)
            )

            bboxes = bboxes / scale
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_h)
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.coco_catIds[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].tolist(),
                    "score": scores[ind].item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self):
        """ generate prediction coco json file """
        print('Evaluate in main process...')
        # write result to coco json format

        t = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.data_list, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What():{}".format(str(e)))
        else:
            f.close()
            if not self.data_list:
                self.file_path = ''
                return self.file_path

            self.data_list.clear()
            return self.file_path

    def get_eval_result(self):
        """Get eval result"""
        if not self.file_path:
            return None, None

        cocoGt = self._coco
        cocoDt = cocoGt.loadRes(self.file_path)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        cocoEval.summarize()
        sys.stdout = stdout
        return rdct.content, cocoEval.stats[0]
