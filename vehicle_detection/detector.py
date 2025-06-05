import torch
import cv2
import numpy as np
from pathlib import Path

# Asegúrate que estas rutas estén bien en tu proyecto
from yolo_core.models.yolo import Model
from yolo_core.utils.datasets import letterbox
from yolo_core.utils.general import non_max_suppression, scale_coords


class VehicleDetector:
    """A class for detecting vehicles in images using a pre-trained YOLOv7 model."""

    def __init__(self, model_path, config_path, nc=3, device=None, img_size=640):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
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
            model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

        model.eval()
        return model

    def detect(self, image):
        """
        Detect vehicles in the given image (expects a BGR NumPy array).
        Returns: list of dicts with 'box', 'conf', and 'class'.
        """
        img0 = image.copy()
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        results = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    results.append({
                        'box': [int(x.item()) for x in xyxy],
                        'conf': float(conf.item()),
                        'class': int(cls.item())
                    })
        return results