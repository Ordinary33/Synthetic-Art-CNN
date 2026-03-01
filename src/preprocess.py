import numpy as np
from PIL import Image

MEAN = MEAN = np.array([0.4843, 0.4340, 0.3911], dtype=np.float32)
STD = np.array([0.2415, 0.2331, 0.2263], dtype=np.float32)
TARGET_SIZE = 256


def preprocess_image(pil_image: Image.Image):
    """
    Matches torchvision.Compose([CenterCrop(256), ToTensor(), Normalize(mean, std)])
    """
    img = pil_image.convert("RGB")

    w, h = img.size
    if w > h:
        new_h = TARGET_SIZE
        new_w = int(w * (TARGET_SIZE / h))

    else:
        new_w = TARGET_SIZE
        new_h = int(h * (TARGET_SIZE / w))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - TARGET_SIZE) / 2
    top = (new_h - TARGET_SIZE) / 2
    img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - MEAN) / STD
    img_np = img_np.transpose(2, 0, 1)

    return np.expand_dims(img_np, axis=0)
