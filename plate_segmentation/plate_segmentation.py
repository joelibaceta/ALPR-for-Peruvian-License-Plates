import torch
import cv2
import numpy as np
from pathlib import Path

from yolo_core.models.yolo import Model
from yolo_core.utils.datasets import letterbox
from yolo_core.utils.general import non_max_suppression, scale_coords

class PlateSegmenter:
    """Detect and segment license plates from vehicle images."""

    def __init__(self, model_path, config_path, nc=1, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, config_path, nc)

    def _load_model(self, weights_path, cfg_path, nc):
        weights_path = Path(weights_path)
        cfg_path = Path(cfg_path)

        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found at {cfg_path}")

        model = Model(cfg_path, ch=3, nc=nc).to(self.device)
        ckpt = torch.load(weights_path, map_location=self.device)
        
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model'].float().state_dict()
        else:
            state_dict = ckpt  # es un state_dict plano

        model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model

    def segment(self, image):
        """Segment license plates in the given vehicle image (BGR NumPy array)."""
        img0 = image.copy()
        img = letterbox(img0, new_shape=640)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        plates = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    crop = img0[y1:y2, x1:x2]
                    plates.append({
                        'box': [x1, y1, x2, y2],
                        'conf': float(conf.item()),
                        'class': int(cls.item()),
                        'crop': crop
                    })
        return plates