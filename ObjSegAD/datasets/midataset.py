import os
import numpy as np
import torch
from mvtec_dataset import MVTecADDataset
import matplotlib.pyplot as plt

# ———— 配置 ————
root_dir = '/ai/dong2/mvtec/data/MVTec'          # MVTec 数据集根目录
object_name = 'hazelnut'                         # 目标类别
anomaly_source_dir = '/ai/dong2/mvtec/data/dtd'   # 纹理目录，用于生成伪缺陷
save_dir = 'result/mid'                          # 保存路径

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

# 实例化 Dataset
ds = MVTecADDataset(root_dir, object_name, mode='train',
                   resize=256, anomaly_source_dir=anomaly_source_dir)

# 原始 train/good 样本数
N = len(ds.paths)

# 三类样本的索引
examples = [
    (0, 'normal'),
    (N, 'background_anomaly'),
    (2 * N, 'structural_defect')
]

# 反归一化参数
mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

# 为每个类别创建子文件夹并保存图片
for idx, class_name in examples:
    # 创建类别子文件夹
    class_dir = os.path.join(save_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    img_in, img_gt, mask = ds[idx]
    
    # 反归一化到 [0,1]
    img_in = (img_in * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img_gt = (img_gt * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()  # [H,W]
    
    # 保存原图
    plt.imsave(os.path.join(class_dir, f'{idx}_original.png'), img_gt)
    
    # 保存伪缺陷图
    plt.imsave(os.path.join(class_dir, f'{idx}_pseudo_defect.png'), img_in)
    
    # 保存掩码（单通道灰度图）
    plt.imsave(os.path.join(class_dir, f'{idx}_mask.png'), mask, cmap='gray')

# 处理整个训练集
for i in range(len(ds)):
    img_in, img_gt, mask = ds[i]
    
    # 确定样本所属类别
    if i < N:
        class_name = 'normal'
    elif i < 2 * N:
        class_name = 'background_anomaly'
    else:
        class_name = 'structural_defect'
    
    # 创建类别子文件夹
    class_dir = os.path.join(save_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # 反归一化到 [0,1]
    img_in = (img_in * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img_gt = (img_gt * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()  # [H,W]
    
    # 保存原图
    plt.imsave(os.path.join(class_dir, f'{i}_original.png'), img_gt)
    
    # 保存伪缺陷图
    plt.imsave(os.path.join(class_dir, f'{i}_pseudo_defect.png'), img_in)
    
    # 保存掩码（单通道灰度图）
    plt.imsave(os.path.join(class_dir, f'{i}_mask.png'), mask, cmap='gray')