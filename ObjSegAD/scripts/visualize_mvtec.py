import os
import argparse
import torch
from torch.utils.data import DataLoader
from mvtec_dataset import MVTecADDataset
from model import ResNet50_DRAEM
from torchvision.utils import save_image


def unnormalize(img_tensor):
    """
    将归一化后的张量反归一化到 [0,1] 区间，用于可视化。
    img_tensor: [3, H, W]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MVTecAD 中间结果可视化并保存")
    parser.add_argument('--data_root', type=str, required=True,
                        help='MVTec AD 数据集根目录，例如 /root/dong_AD/MVTec/data/mvtec')
    parser.add_argument('--object_name', type=str, required=True,
                        help='类别名称，比如 zipper、bottle 等')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重 (.pth) 路径')
    parser.add_argument('--resize', type=int, default=256,
                        help='输入图像尺寸 (h=w=resize)')
    parser.add_argument('--anomaly_source_dir', type=str, default=None,
                        help='伪缺陷纹理目录，例如 /root/dong_AD/MVTec/data/dtd')
    parser.add_argument('--output_dir', type=str, default='output_mvtec',
                        help='结果保存根目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='要可视化的样本数量')
    args = parser.parse_args()

    # 创建输出文件夹
    base = os.path.join(args.output_dir, args.object_name)
    pseudo_dir = os.path.join(base, 'pseudo_defect')
    recon_dir  = os.path.join(base, 'recon_branch')
    seg_dir    = os.path.join(base, 'seg_branch')
    for d in (pseudo_dir, recon_dir, seg_dir):
        os.makedirs(d, exist_ok=True)

    # 加载训练模式数据集以获取伪缺陷合成
    dataset = MVTecADDataset(root_dir=args.data_root,
                             object_name=args.object_name,
                             mode='train',
                             resize=args.resize,
                             anomaly_source_dir=args.anomaly_source_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50_DRAEM().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 逐样本推理并保存
    for idx, (img_in, img_gt, mask_gt) in enumerate(loader):
        if idx >= args.num_samples:
            break

        img_in = img_in.to(device)
        with torch.no_grad():
            out_rec, out_seg = model(img_in)
            seg_prob = torch.sigmoid(out_seg)

        # 1) 保存伪缺陷合成结果（输入带伪缺陷的图像）
        inp_vis = unnormalize(img_in[0].cpu())
        save_image(inp_vis, os.path.join(pseudo_dir, f'{idx:04d}.png'))

        # 2) 保存重构分支输出
        rec_vis = unnormalize(out_rec[0].cpu())
        save_image(rec_vis, os.path.join(recon_dir, f'{idx:04d}.png'))

        # 3) 保存分割分支输出（异常概率图）
        save_image(seg_prob[0].cpu(), os.path.join(seg_dir, f'{idx:04d}.png'))

    print(f"已保存 {min(len(dataset), args.num_samples)} 个样本的中间结果到 '{base}' 下的 pseudo_defect, recon_branch, seg_branch。")

    # 应用示例：
    # python visualize_mvtec.py --data_root /root/dong_AD/MVTec/data/mvtec \
    #     --object_name zipper \
    #     --model_path zipper_best_img.pth \
    #     --anomaly_source_dir /root/dong_AD/MVTec/data/dtd \
    #     --output_dir vis_results \
    #     --num_samples 5
