import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

def load_image(path, resize):
    image = Image.open(path).convert("RGB").resize((resize, resize))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    return transform(image).unsqueeze(0)

def post_process(pred, h, w):
    pred = pred.squeeze().cpu().data.numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = (pred * 255).astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
