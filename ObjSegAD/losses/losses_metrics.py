# loss_metric.py
# 统一的损失函数和评估指标模块，支持多个数据集

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn import metrics
from typing import List, Dict, Tuple, Optional, Union

# ============================================================================
# 损失函数类
# ============================================================================

class FocalLoss(nn.Module):
    """焦点损失，用于处理类别不平衡问题"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算焦点损失
        Args:
            logits: 模型输出的logits [B, C, H, W] 或 [B, C]
            targets: 目标标签 [B, H, W] 或 [B]
        """
        if logits.dim() == 4:  # 像素级
            return self._pixel_focal_loss(logits, targets)
        else:  # 图像级
            return self._image_focal_loss(logits, targets)
    
    def _pixel_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """像素级焦点损失"""
        eps = 1e-6
        pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
        
        if targets.dim() == 3:  # [B, H, W]
            targets = targets.unsqueeze(1)  # [B, 1, H, W]
        
        pt = torch.where(targets == 1, pred, 1 - pred)
        log_pt = torch.log(pt)
        weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = -alpha_t * weight * log_pt
        return loss.mean()
    
    def _image_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """图像级焦点损失"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class SSIMLoss(nn.Module):
    """结构相似性损失"""
    
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算SSIM损失
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        Returns:
            1 - SSIM值
        """
        ssim_val = self._ssim_index(pred, target)
        return 1.0 - ssim_val
    
    def _ssim_index(self, img_pred: torch.Tensor, img_true: torch.Tensor) -> torch.Tensor:
        """计算SSIM指数"""
        if img_pred.dim() == 4:
            ssim_vals = [self._ssim_index(img_pred[i], img_true[i]) for i in range(img_pred.size(0))]
            return sum(ssim_vals) / len(ssim_vals)
        
        if img_pred.size(0) == 3:
            return (self._ssim_index(img_pred[0:1], img_true[0:1]) +
                   self._ssim_index(img_pred[1:2], img_true[1:2]) +
                   self._ssim_index(img_pred[2:3], img_true[2:3])) / 3.0
        
        x, y = img_pred, img_true
        C1, C2 = 0.01**2, 0.03**2
        kernel_size = self.window_size
        pad = kernel_size // 2
        window = torch.ones((1, 1, kernel_size, kernel_size), device=x.device) / (kernel_size * kernel_size)
        
        mu_x = F.conv2d(x.unsqueeze(0), window, padding=pad)[0]
        mu_y = F.conv2d(y.unsqueeze(0), window, padding=pad)[0]
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x.unsqueeze(0)*x.unsqueeze(0), window, padding=pad)[0] - mu_x_sq
        sigma_y_sq = F.conv2d(y.unsqueeze(0)*y.unsqueeze(0), window, padding=pad)[0] - mu_y_sq
        sigma_xy = F.conv2d(x.unsqueeze(0)*y.unsqueeze(0), window, padding=pad)[0] - mu_xy
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return ssim_map.mean()


class ReconstructionLoss(nn.Module):
    """重构损失：MSE + (1 - SSIM)"""
    
    def __init__(self, ssim_weight: float = 1.0):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred_recon: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred_recon, target_img)
        ssim_loss = self.ssim_loss(pred_recon.clamp(0, 1), target_img.clamp(0, 1))
        return mse + self.ssim_weight * ssim_loss


# ============================================================================
# 评估指标函数
# ============================================================================

def compute_aupro(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], 
                  max_fpr: float = 0.3, num_thresholds: int = 100) -> float:
    """
    计算像素级AUPRO (Area Under Per-Region Overlap)
    
    Args:
        pred_masks: 预测异常概率图列表，每项形状[H,W]，值域[0,1]
        gt_masks: 真值缺陷掩码列表，每项形状[H,W]，值为0或1
        max_fpr: 最大假阳性率阈值
        num_thresholds: 评估阈值数量
    
    Returns:
        AUPRO值
    """
    region_hit_list = []
    total_neg_pixels = 0
    
    for pm, gt in zip(pred_masks, gt_masks):
        total_neg_pixels += np.sum(gt == 0)
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 0:
                mask_region = np.zeros_like(gt, dtype=np.uint8)
                cv2.drawContours(mask_region, [cnt], -1, 1, thickness=-1)
                mask_bool = mask_region.astype(bool)
                region_hit_list.append((mask_bool, pm))
    
    thresholds = np.linspace(1.0, 0.0, num_thresholds)
    pros, fprs = [], []
    
    for thr in thresholds:
        # 计算FPR
        fp = sum(np.sum((pm > thr) & (gt == 0)) for pm, gt in zip(pred_masks, gt_masks))
        fpr = fp / float(total_neg_pixels) if total_neg_pixels > 0 else 0.0
        fprs.append(fpr)
        
        # 计算PRO
        recalls = [1.0 if np.any(pm[mask_bool] > thr) else 0.0 
                  for mask_bool, pm in region_hit_list]
        pro = np.mean(recalls) if recalls else 0.0
        pros.append(pro)
    
    # 计算AUPRO
    fprs, pros = np.array(fprs), np.array(pros)
    valid_idx = fprs <= max_fpr
    if not np.any(valid_idx):
        return 0.0
    
    fprs_valid = fprs[valid_idx]
    pros_valid = pros[valid_idx]
    
    if len(fprs_valid) < 2:
        return 0.0
    
    aupro = np.trapz(pros_valid, fprs_valid) / max_fpr
    return aupro


def compute_metrics(y_true_img: np.ndarray, y_score_img: np.ndarray,
                   y_true_pix: np.ndarray, y_score_pix: np.ndarray,
                   masks_gt: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
    """
    计算综合评估指标
    
    Args:
        y_true_img: 图像级真值标签 [N]
        y_score_img: 图像级预测分数 [N]
        y_true_pix: 像素级真值标签 [N*H*W]
        y_score_pix: 像素级预测分数 [N*H*W]
        masks_gt: 用于AUPRO计算的GT掩码列表
    
    Returns:
        包含各种指标的字典
    """
    result = {}
    
    # 图像级指标
    if len(y_true_img) > 0 and len(np.unique(y_true_img)) > 1:
        result['img_auc'] = metrics.roc_auc_score(y_true_img, y_score_img)
        result['img_ap'] = metrics.average_precision_score(y_true_img, y_score_img)
    
    # 像素级指标
    if len(y_true_pix) > 0 and len(np.unique(y_true_pix)) > 1:
        result['pix_auc'] = metrics.roc_auc_score(y_true_pix, y_score_pix)
        result['pix_ap'] = metrics.average_precision_score(y_true_pix, y_score_pix)
        
        # 计算最优F1分数
        precision, recall, thresholds = metrics.precision_recall_curve(y_true_pix, y_score_pix)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        result['pix_f1_max'] = np.max(f1_scores)
    
    # AUPRO指标（如果提供了masks_gt）
    if masks_gt is not None:
        # 将像素级预测重新组织为图像列表
        pred_masks = []
        gt_masks = []
        
        # 假设输入是展平的，需要根据实际情况调整
        # 这里提供一个通用的接口，具体实现可能需要根据数据格式调整
        if hasattr(masks_gt, '__len__') and len(masks_gt) > 0:
            try:
                result['aupro'] = compute_aupro(pred_masks, masks_gt)
            except Exception as e:
                print(f"AUPRO计算失败: {e}")
    
    return result


# ============================================================================
# 统一的损失和指标管理器
# ============================================================================

class UnifiedLossMetric:
    """
    统一的损失函数和评估指标管理器
    支持不同数据集的特定配置
    """
    
    def __init__(self, dataset: str = 'general'):
        self.dataset = dataset
        self.loss_configs = self._get_loss_configs()
        self.metric_configs = self._get_metric_configs()
        
        # 初始化损失函数
        self.focal_loss = FocalLoss(**self.loss_configs['focal'])
        self.ssim_loss = SSIMLoss(**self.loss_configs['ssim'])
        self.recon_loss = ReconstructionLoss(**self.loss_configs['reconstruction'])
    
    def _get_loss_configs(self) -> Dict:
        """获取数据集特定的损失函数配置"""
        configs = {
            'btad': {
                'focal': {'alpha': 0.25, 'gamma': 2.0},
                'ssim': {'window_size': 11},
                'reconstruction': {'ssim_weight': 1.0}
            },
            'dagm': {
                'focal': {'alpha': 0.3, 'gamma': 2.5},
                'ssim': {'window_size': 11},
                'reconstruction': {'ssim_weight': 1.2}
            },
            'visa': {
                'focal': {'alpha': 0.25, 'gamma': 2.0},
                'ssim': {'window_size': 11},
                'reconstruction': {'ssim_weight': 1.0}
            },
            'mpdd': {
                'focal': {'alpha': 0.7, 'gamma': 2.0},
                'ssim': {'window_size': 11},
                'reconstruction': {'ssim_weight': 1.0}
            },
            'mvtec': {
                'focal': {'alpha': 0.25, 'gamma': 2.0},
                'ssim': {'window_size': 11},
                'reconstruction': {'ssim_weight': 1.0}
            }
        }
        return configs.get(self.dataset, configs['btad'])
    
    def _get_metric_configs(self) -> Dict:
        """获取数据集特定的评估指标配置"""
        configs = {
            'btad': {'aupro_max_fpr': 0.3, 'aupro_thresholds': 100},
            'dagm': {'aupro_max_fpr': 0.3, 'aupro_thresholds': 100},
            'visa': {'aupro_max_fpr': 0.3, 'aupro_thresholds': 100},
            'mpdd': {'aupro_max_fpr': 0.3, 'aupro_thresholds': 100},
            'mvtec': {'aupro_max_fpr': 0.3, 'aupro_thresholds': 100}
        }
        return configs.get(self.dataset, configs['btad'])
    
    def compute_loss(self, pred_dict: Dict[str, torch.Tensor], 
                    target_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Args:
            pred_dict: 预测结果字典，可包含 'anomaly_map', 'reconstruction', 'classification'
            target_dict: 目标字典，可包含 'anomaly_mask', 'image', 'label'
        
        Returns:
            损失字典
        """
        losses = {}
        
        # 异常分割损失
        if 'anomaly_map' in pred_dict and 'anomaly_mask' in target_dict:
            losses['focal'] = self.focal_loss(pred_dict['anomaly_map'], target_dict['anomaly_mask'])
        
        # 重构损失
        if 'reconstruction' in pred_dict and 'image' in target_dict:
            losses['reconstruction'] = self.recon_loss(pred_dict['reconstruction'], target_dict['image'])
        
        # 分类损失
        if 'classification' in pred_dict and 'label' in target_dict:
            losses['classification'] = self.focal_loss(pred_dict['classification'], target_dict['label'])
        
        # 总损失
        if losses:
            losses['total'] = sum(losses.values())
        
        return losses
    
    def compute_metrics(self, pred_dict: Dict[str, np.ndarray], 
                       target_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            pred_dict: 预测结果字典
            target_dict: 目标字典
        
        Returns:
            指标字典
        """
        y_true_img = target_dict.get('img_labels', np.array([]))
        y_score_img = pred_dict.get('img_scores', np.array([]))
        y_true_pix = target_dict.get('pix_labels', np.array([]))
        y_score_pix = pred_dict.get('pix_scores', np.array([]))
        masks_gt = target_dict.get('masks_gt', None)
        
        return compute_metrics(y_true_img, y_score_img, y_true_pix, y_score_pix, masks_gt)


# ============================================================================
# 向后兼容函数
# ============================================================================

def ssim_index(img_pred: torch.Tensor, img_true: torch.Tensor) -> torch.Tensor:
    """向后兼容的SSIM计算函数"""
    ssim_loss = SSIMLoss()
    return 1.0 - ssim_loss(img_pred, img_true)


def reconstruction_loss(pred_recon: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
    """向后兼容的重构损失函数"""
    recon_loss = ReconstructionLoss()
    return recon_loss(pred_recon, target_img)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, 
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """向后兼容的焦点损失函数"""
    focal = FocalLoss(alpha, gamma)
    return focal(logits, targets)


# ============================================================================
# 主函数和命令行接口
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='统一损失函数和评估指标模块')
    parser.add_argument('--dataset', type=str, default='general',
                       choices=['btad', 'dagm', 'visa', 'mpdd', 'mvtec', 'general'],
                       help='数据集名称')
    parser.add_argument('--test', action='store_true', help='运行测试')
    
    args = parser.parse_args()
    
    if args.test:
        print(f"测试数据集 {args.dataset} 的损失函数和指标...")
        
        # 创建管理器
        manager = UnifiedLossMetric(args.dataset)
        
        # 测试损失函数
        pred_dict = {
            'anomaly_map': torch.randn(2, 1, 64, 64),
            'reconstruction': torch.randn(2, 3, 64, 64),
            'classification': torch.randn(2, 2)
        }
        
        target_dict = {
            'anomaly_mask': torch.randint(0, 2, (2, 64, 64)),
            'image': torch.randn(2, 3, 64, 64),
            'label': torch.randint(0, 2, (2,))
        }
        
        losses = manager.compute_loss(pred_dict, target_dict)
        print("损失函数测试结果:")
        for name, loss in losses.items():
            print(f"  {name}: {loss.item():.4f}")
        
        # 测试评估指标
        pred_metrics = {
            'img_scores': np.random.rand(100),
            'pix_scores': np.random.rand(100 * 64 * 64)
        }
        
        target_metrics = {
            'img_labels': np.random.randint(0, 2, 100),
            'pix_labels': np.random.randint(0, 2, 100 * 64 * 64)
        }
        
        metrics_result = manager.compute_metrics(pred_metrics, target_metrics)
        print("\n评估指标测试结果:")
        for name, value in metrics_result.items():
            print(f"  {name}: {value:.4f}")
        
        print(f"\n{args.dataset} 数据集的损失函数和指标测试完成！")
    else:
        print("统一损失函数和评估指标模块已加载")
        print("支持的数据集:", ['btad', 'dagm', 'visa', 'mpdd', 'mvtec'])
        print("使用 --test 参数运行测试")