import os
import sys
import cv2
import numpy as np
import mindspore
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

# Add 'yolox' directory to path to allow imports from yolox package and datasets
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolox'))

from yolox.models.yolox import YOLOX
from yolox.exp.build import get_exp
from yolox.config import config
from datasets.transform import ValTransform

class YOLOXInfer:
    def __init__(self, ckpt_path, device_target="CPU", name="yolox-tiny"):
        """
        Initialize the YOLOX inference model.
        
        Args:
            ckpt_path (str): Path to the trained checkpoint file.
            device_target (str): "CPU" (default), "GPU", or "Ascend".
            name (str): Model name (e.g. "yolox-tiny").
        """
        self.config = config
        self.device_target = device_target
        
        # Set context
        context.set_context(mode=context.GRAPH_MODE, device_target=self.device_target, save_graphs=False)
        
        # Build model
        self.exp = get_exp(name)
        # Using "yolopafpn" backbone to match training configuration
        self.network = YOLOX(self.config, backbone="yolopafpn", exp=self.exp) 
        self.network.set_train(False)
        
        # Load checkpoint
        print(f"Loading checkpoint from {ckpt_path}...")
        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(self.network, param_dict)
        
        # Preprocessing tool
        self.preproc = ValTransform(legacy=False)

    def predict(self, image_path, conf_thre=0.1, nms_thre=0.45):
        """
        Run inference on a single image.
        
        Args:
            image_path (str): Path to the input image.
            conf_thre (float): Confidence threshold.
            nms_thre (float): NMS threshold.
            
        Returns:
            list: List of detections. Each detection is a dict with:
                  'bbox': [x1, y1, x2, y2],
                  'score': float,
                  'class_id': int,
                  'class_name': str (if available, otherwise "Class <id>")
        """
        # 1. Read Image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # 2. Preprocess
        tensor_img, _ = self.preproc(img, self.config.input_size)
        tensor_img = np.expand_dims(tensor_img, axis=0)
        tensor_img = Tensor(tensor_img, mindspore.float32)

        # 3. Model Inference
        outputs = self.network(tensor_img)
        outputs = outputs.asnumpy()

        # 4. Postprocess
        # outputs shape: (batch_size, 8400, 85)
        post_outputs = self.postprocess(
            outputs, 
            self.config.num_classes, 
            conf_thre, 
            nms_thre
        )
        
        # 5. Format Results
        # post_outputs[0] is array of [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
        results = []
        prediction = post_outputs[0]
        
        if prediction is None:
            return []

        # Resize boxes back to original image size
        img_h, img_w = img.shape[:2]
        scale = min(self.config.input_size[0] / img_h, self.config.input_size[1] / img_w)
        
        for det in prediction:
            x1, y1, x2, y2 = det[:4] / scale
            obj_conf = det[4]
            class_conf = det[5]
            class_id = int(det[6])
            
            score = obj_conf * class_conf
            
            # Simple result structure
            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
                "class_id": class_id,
            })
            
        return results

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
            class_conf = np.max(image_pred[:, 5:5 + num_classes], axis=-1)
            class_pred = np.argmax(image_pred[:, 5:5 + num_classes], axis=-1)
            conf_mask = (image_pred[:, 4] * class_conf >= conf_thre).squeeze()
            class_conf = np.expand_dims(class_conf, axis=-1)
            class_pred = np.expand_dims(class_pred, axis=-1).astype(np.float16)
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
        """Calculate Nms based on class info,Each index value correspond to a category"""
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

    def visualize(self, image_path, results, save_path="prediction.jpg"):
        """
        Draw bounding boxes on the image and save it.
        """
        img = cv2.imread(image_path)
        for res in results:
            bbox = res["bbox"]
            score = res["score"]
            cls_id = res["class_id"]
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            text = f"ID:{cls_id} {score:.2f}"
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        cv2.imwrite(save_path, img)
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # Example Usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--device_target", type=str, default="CPU", help="Device target")
    parser.add_argument("--backbone", type=str, default="yolopafpn", help="Backbone type")
    
    # We parse known args to avoid conflict if other args are passed implicitly by config parser
    args, unknown = parser.parse_known_args()

    model = YOLOXInfer(ckpt_path=args.ckpt_path, device_target=args.device_target)
    results = model.predict(args.image)
    print(f"Detected {len(results)} objects:")
    for r in results:
        print(r)
    
    model.visualize(args.image, results)
