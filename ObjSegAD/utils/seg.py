import os
import torch
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image

# -----------------------------------------------------------------------------
# 配置：本地权重文件路径，先用 wget 下载到该位置：
#   wget https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth \
#        -O models/deeplabv3_resnet50_coco.pth
# -----------------------------------------------------------------------------
WEIGHTS_PATH = "models/deeplabv3_resnet50_coco.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1. 初始化模型：开启 aux classifier，num_classes=21 与预训练权重一致
# -----------------------------------------------------------------------------
model = deeplabv3_resnet50(pretrained=False,
                           num_classes=21,
                           aux_loss=True)        # <<-- 这一行很关键
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.to(DEVICE).eval()

# -----------------------------------------------------------------------------
# 2. 定义与 COCO 预训练一致的图像预处理
# -----------------------------------------------------------------------------
preprocess = T.Compose([
    T.Resize((256, 256)),  # 根据项目需求调整
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------------------
# 3. 批量生成前景掩码
# -----------------------------------------------------------------------------
dataset_root = "/ai/dong2/mvtec/data/MVTec"
for cls in os.listdir(dataset_root):
    train_dir = os.path.join(dataset_root, cls, "train", "good")
    if not os.path.isdir(train_dir):
        continue

    mask_dir = os.path.join(dataset_root, cls, "train_mask", "good")
    os.makedirs(mask_dir, exist_ok=True)

    for fname in os.listdir(train_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        img_path = os.path.join(train_dir, fname)
        img = Image.open(img_path).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(inp)["out"][0]          # 21 x H x W
            pred = out.argmax(0).cpu().numpy()  # H x W

        # 非类别0视为前景
        fg_mask = (pred != 0).astype("uint8") * 255

        # 保存掩码
        base, ext = os.path.splitext(fname)
        mask_name = f"{base}_mask{ext}"
        mask_path = os.path.join(mask_dir, mask_name)
        Image.fromarray(fg_mask, mode="L").save(mask_path)
        print(f"Saved mask: {mask_path}")
