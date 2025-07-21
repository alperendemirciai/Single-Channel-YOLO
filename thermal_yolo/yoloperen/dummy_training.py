import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import yaml

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)



# ====================== STEP 1: Custom DataLoader ======================

def build_dataloader_from_custom_yolo_dataset(img_path, batch_size=4, augment=False, shuffle=False, prefix="data: "):

    data_dict = load_yaml("dummy_dataset/data.yaml")
    dataset = YOLODataset(
        img_path=img_path,
        imgsz=640,
        augment=augment,
        rect=False,
        cache=False,
        prefix=prefix,
        data=data_dict
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
    return loader


# ====================== STEP 2: Training, Validation, and Testing ======================

def train_validate_test_yolo():
    yaml_path = "dummy_dataset/data.yaml"
    model = YOLO("models/customyolo11n.yaml")  # Replace with your custom YOLO model config

    # Disable HSV augmentations (grayscale input)
    model.overrides.update(dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0))

    # Train
    model.train(data=yaml_path, epochs=5, imgsz=640, batch=4)

    # Validate
    model.val(data=yaml_path)

    # Test inference
    test_loader = build_dataloader_from_custom_yolo_dataset("dummy_dataset/images/test", batch_size=1)
    model.eval()

    print("🧪 Running test inference...")
    for imgs, targets, paths, _ in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(model.device, dtype=torch.float32)
        preds = model(imgs, verbose=False)

        for path, pred in zip(paths, preds):
            print(f"Image: {path}, Detected: {len(pred.boxes)} objects")


if __name__ == "__main__":
    train_validate_test_yolo()
