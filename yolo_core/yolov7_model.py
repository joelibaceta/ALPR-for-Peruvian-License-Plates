from models.yolo import Model
import torch
import os

def load_model(weight_path, cfg_path='yolo_core/cfg/yolov7.yaml', nc=3, device='cpu'):
    # 1. Cargar arquitectura
    model = Model(cfg_path, ch=3, nc=nc).to(device)

    # 2. Cargar pesos
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    model.eval()
    
    return model