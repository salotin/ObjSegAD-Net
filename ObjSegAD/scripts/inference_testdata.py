import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import ResNet50_DRAEM  # 确保 model.py 可用
import argparse

def load_image(path, resize=256):
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0), image

def save_heatmap(mask, orig_img, save_path):
    # mask: [H, W] 归一化
    plt.figure(figsize=(6, 6))
    plt.imshow(orig_img)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@torch.no_grad()
def run_inference(model_path, test_dir, output_dir, resize=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50_DRAEM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    image_list = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_list:
        img_path = os.path.join(test_dir, img_name)
        img_tensor, img_pil = load_image(img_path, resize)
        img_tensor = img_tensor.to(device)

        _, seg_logits = model(img_tensor)
        prob_map = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)

        heatmap_save_path = os.path.join(output_dir, img_name.replace('.', '_heatmap.'))
        save_heatmap(prob_map, img_pil.resize((resize, resize)), heatmap_save_path)
        print(f"Inference done for {img_name} -> saved to {heatmap_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='路径到训练好的模型（如 capsule_best_pix.pth）')
    parser.add_argument('--test_dir', type=str, default='./testdata', help='测试图像路径（无掩码）')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='热图保存路径')
    parser.add_argument('--resize', type=int, default=256, help='图像缩放大小')
    args = parser.parse_args()

    run_inference(args.model_path, args.test_dir, args.output_dir, args.resize)
