import os
import numpy as np
from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm
import cv2
import yaml

# Configuration
NUM_IMAGES = 20  # per split
IMAGE_SIZE = (640, 640)  # H x W
NUM_CLASSES = 1
CLASS_NAMES = ["object"]  # Update with real class names if needed
DATA_DIR = Path("dummy_dataset")
SPLITS = ["train", "val", "test"]
BBOX_PER_IMAGE = 3  # bounding boxes per image

def create_dirs():
    for split in SPLITS:
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def generate_16bit_image(size):
    """Create a single-channel 16-bit image with random noise"""
    return (np.random.rand(*size) * 65535).astype(np.uint16)

def generate_random_bboxes(image_shape, num_boxes=3):
    h, w = image_shape
    boxes = []
    for _ in range(num_boxes):
        bw = random.uniform(0.1, 0.4)
        bh = random.uniform(0.1, 0.4)
        cx = random.uniform(bw / 2, 1 - bw / 2)
        cy = random.uniform(bh / 2, 1 - bh / 2)
        boxes.append([0, cx, cy, bw, bh])
    return boxes

def save_image(image_array, path):
    img = Image.fromarray(image_array, mode="I;16")
    img.save(path)

def save_labels(bboxes, path):
    with open(path, "w") as f:
        for box in bboxes:
            f.write(" ".join(f"{v:.6f}" for v in box) + "\n")

def write_data_yaml():
    data_yaml = {
        'train': str(DATA_DIR / "images" / "train"),
        'val': str(DATA_DIR / "images" / "val"),
        'test': str(DATA_DIR / "images" / "test"),
        'nc': NUM_CLASSES,
        'names': CLASS_NAMES
    }
    with open(DATA_DIR / "data.yaml", "w") as f:
        yaml.dump(data_yaml)
    print("✅ data.yaml written to", DATA_DIR / "data.yaml")

def test_image_properties():
    print("🔍 Verifying image properties...")
    sample_path = DATA_DIR / "images" / "train"
    sample_file = next(sample_path.glob("*.png"))

    img = cv2.imread(str(sample_file), cv2.IMREAD_UNCHANGED)
    print(f"Sample image: {sample_file.name}")
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    if img.ndim == 2 and img.dtype == np.uint16:
        print("✅ Image is single-channel 16-bit.")
    else:
        print("❌ Image is not 1-channel 16-bit.")

def generate_dataset():
    create_dirs()
    for split in SPLITS:
        print(f"Generating {split} data...")
        for i in tqdm(range(NUM_IMAGES)):
            img_name = f"{split}_{i:04d}.png"
            label_name = img_name.replace(".png", ".txt")

            img = generate_16bit_image(IMAGE_SIZE)
            bboxes = generate_random_bboxes(IMAGE_SIZE)

            save_image(img, DATA_DIR / "images" / split / img_name)
            save_labels(bboxes, DATA_DIR / "labels" / split / label_name)

    write_data_yaml()
    test_image_properties()

if __name__ == "__main__":
    generate_dataset()
    print("✅ Dummy 16-bit grayscale PNG dataset created in 'dummy_dataset/'")
