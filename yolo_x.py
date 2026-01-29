import os
import sys
import numpy as np
import cv2
import argparse
from typing import Union, List, Optional

# Ensure we can import from inference.py and yolox package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from inference import YOLOXInfer
    import mindspore
    from mindspore import Tensor
except ImportError:
    print("Error: Could not import YOLOXInfer from inference.py. Make sure this script is in the same directory.")
    sys.exit(1)

# mimics Ultralytics YOLOv8/v11/v12 Boxes object
class Boxes:
    def __init__(self, boxes, conf, cls, orig_shape):
        """
        boxes: numpy array [N, 4] (xyxy)
        conf: numpy array [N]
        cls: numpy array [N]
        orig_shape: (h, w)
        """
        self.orig_shape = orig_shape
        self.data = np.concatenate([boxes, conf[:, None], cls[:, None]], axis=1) if len(boxes) else np.zeros((0, 6))
        self.xyxy = boxes if len(boxes) else np.zeros((0, 4))
        self.conf = conf if len(conf) else np.zeros((0,))
        self.cls = cls if len(cls) else np.zeros((0,))
        
        # xywh (center x, center y, width, height)
        if len(boxes):
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            cx = boxes[:, 0] + w / 2
            cy = boxes[:, 1] + h / 2
            self.xywh = np.stack([cx, cy, w, h], axis=1)
        else:
            self.xywh = np.zeros((0, 4))

    def cpu(self):
        return self
    
    def numpy(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Boxes(
                self.xyxy[idx][None],
                self.conf[idx][None],
                self.cls[idx][None],
                self.orig_shape
            )
        return Boxes(
            self.xyxy[idx],
            self.conf[idx],
            self.cls[idx],
            self.orig_shape
        )

# mimics Ultralytics YOLOv8/v11/v12 Results object
class Results:
    def __init__(self, orig_img, path, names, boxes):
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = path
        self.names = names # {0: 'person', 1: 'bicycle', ...}
        self.boxes = boxes # Boxes object
        self.probs = None # No classification only
        self.masks = None # No segmentation
        self.keypoints = None # No pose
        self.obb = None

    def __len__(self):
        return len(self.boxes)

    def plot(self, conf=True, line_width=None, font_size=None, font='Arial.ttf', labels=True, boxes=True, color_mode='class', alpha=1.0):
        """
        Plots predictions on the image. Mimics ultralytics plot()
        Returns: numpy array of the image with drawn boxes
        """
        img = self.orig_img.copy()
        if not boxes:
            return img
            
        for i in range(len(self.boxes)):
            box = self.boxes.xyxy[i].astype(int)
            c = int(self.boxes.cls[i])
            score = self.boxes.conf[i]
            
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            # Create a unique color per class
            np.random.seed(c)
            color = np.random.randint(0, 255, size=3).tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            if labels:
                label_text = f"{self.names.get(c, str(c))}"
                if conf:
                    label_text += f" {score:.2f}"
                
                t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), color, -1)
                cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return img

    def save(self, filename="prediction.jpg"):
        res_img = self.plot()
        cv2.imwrite(filename, res_img)
        print(f"Saved visualization to {filename}")

# High level wrapper
class YOLO:
    def __init__(self, model_path: str, device: str = 'CPU', backbone: str = 'yolopafpn'):
        """
        Mimics YOLO('model.pt')
        Args:
            model_path: Checkpoint file path
            device: 'CPU', 'GPU' etc
            backbone: architecture name match
        """
        print(f"Initializing YOLOv12-style wrapper for MindSpore YOLOX...")
        try:
            # We initialize the underlying YOLOXInfer.
            # We must pass the correct backbone to avoid architecture mismatch.
            # However, YOLOXInfer in inference.py doesn't expose all args in __init__ in some versions,
            # but in the recently fixed version it does NOT take backbone in init, it takes name.
            # Wait, our fixed inference.py init is: def __init__(self, ckpt_path, device_target="CPU", name="yolox-tiny")
            # And it hardcoded: self.network = YOLOX(self.config, backbone="yolopafpn"...)
            # So we rely on that.
            self.model = YOLOXInfer(ckpt_path=model_path, device_target=device)
            # Default names dict (0..num_classes)
            self.names = {i: f'class_{i}' for i in range(self.model.config.num_classes)}
        except Exception as e:
            print(f"Failed to initialize YOLOXInfer: {e}")
            raise e

    def __call__(self, source: Union[str, np.ndarray, List[str]], conf: float = 0.25, iou: float = 0.45, verbose: bool = True) -> List[Results]:
        """
        Run inference.
        source: Image path, or numpy array (HWC), or list of paths.
        conf: Confidence threshold.
        iou: NMS threshold.
        
        Returns: A list of Results objects (one per image).
        """
        # Unify source to list
        if isinstance(source, str):
            sources = [source]
        elif isinstance(source, np.ndarray):
            sources = [source]
        elif isinstance(source, list):
            sources = source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        results_list = []

        for p in sources:
            if isinstance(p, str):
                orig_img = cv2.imread(p)
                if orig_img is None:
                    print(f"Warning: Could not read image {p}")
                    continue
                path = p
            elif isinstance(p, np.ndarray):
                orig_img = p
                path = "numpy_array"
            else:
                continue

            # PREPROCESS (Mimic logic from inference.py but decoupled to handle numpy)
            # The fixed inference.py uses:
            # tensor_img, _ = self.preproc(img, self.config.input_size)
            # Since self.model is YOLOXInfer, we can access .preproc and .config
            
            img_h, img_w = orig_img.shape[:2]
            tensor_img, _ = self.model.preproc(orig_img, self.model.config.input_size)
            tensor_img = np.expand_dims(tensor_img, axis=0)
            tensor_img = Tensor(tensor_img, mindspore.float32)

            # INFERENCE
            outputs = self.model.network(tensor_img)
            outputs = outputs.asnumpy()

            # POSTPROCESS (Using the postprocess method from YOLOXInfer)
            post_outputs = self.model.postprocess(
                outputs, 
                self.model.config.num_classes, 
                conf, 
                iou
            )
            
            # FORMAT RESULTS
            prediction = post_outputs[0]
            
            if prediction is None:
                boxes_list = []
                conf_list = []
                cls_list = []
            else:
                # Resize boxes back to original image size
                scale = min(self.model.config.input_size[0] / img_h, self.model.config.input_size[1] / img_w)
                
                # prediction is [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
                # We need final score = obj_conf * class_conf
                
                x1 = prediction[:, 0] / scale
                y1 = prediction[:, 1] / scale
                x2 = prediction[:, 2] / scale
                y2 = prediction[:, 3] / scale
                
                bboxes = np.stack([x1, y1, x2, y2], axis=1)
                scores = prediction[:, 4] * prediction[:, 5]
                classes = prediction[:, 6]
                
                boxes_list = bboxes
                conf_list = scores
                cls_list = classes

            # Create standard Results object
            boxes_obj = Boxes(
                boxes=np.array(boxes_list),
                conf=np.array(conf_list),
                cls=np.array(cls_list),
                orig_shape=(img_h, img_w)
            )
            
            result = Results(orig_img, path, self.names, boxes_obj)
            results_list.append(result)
            
            if verbose:
                print(f"image {path}: {img_h}x{img_w} {len(boxes_obj)} objects")

        return results_list

    def names(self):
        return self.names

if __name__ == "__main__":
    # Example CLI mimics: python yolo_v12.py predict model=ckpt source=img
    # But adhering to argparse for simplicity while keeping keys similar
    parser = argparse.ArgumentParser(description="YOLOv12-style inference wrapper for MindSpore YOLOX")
    parser.add_argument("tasks", nargs="*", help="Optional task (predict)") # ignored mostly
    parser.add_argument("--model", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--device", type=str, default="CPU", help="Device (CPU/GPU)")
    parser.add_argument("--save", action="store_true", help="Save prediction image")
    
    # Parse known args to avoid legacy config conflicts if they arise
    args, unknown = parser.parse_known_args()
    
    # Init Model
    yolo = YOLO(model_path=args.model, device=args.device)
    
    # Run Inference
    results = yolo(source=args.source, conf=args.conf, iou=args.iou)
    
    # Display/Save
    for res in results:
        # Access results just like YOLOv8
        # boxes = res.boxes
        # print(boxes.xyxy)
        
        if args.save:
            res.save("prediction_v12.jpg")
