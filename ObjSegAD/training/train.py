# train.py - 统一训练脚本
# 整合了针对不同数据集(BTAD, DAGM, MPDD, MVTec, VisA)的训练功能

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# 导入统一的模型和损失函数
try:
    from model import create_model, ResNet50_DRAEM
except ImportError:
    print("Warning: 无法导入统一模型，尝试导入原始模型")
    from models.model_btad import ResNet50_DRAEM

try:
    from loss_metric import UnifiedLossMetric
except ImportError:
    print("Warning: 无法导入统一损失函数，尝试导入原始损失函数")
    from losses_metrics import reconstruction_loss, focal_loss

# 数据集导入
from datasets_dataset.btad_dataset import BTADDataset
from datasets_dataset.dagm_dataset import DAGMDataset
from datasets_dataset.mpdd_dataset import MPDDDataset
from datasets_dataset.mvtec_dataset import MVTecADDataset
from datasets_dataset.visa_dataset import VisADataset


class UnifiedTrainer:
    """统一训练器类，支持所有数据集的训练"""
    
    def __init__(self, dataset_name, args):
        self.dataset_name = dataset_name.lower()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化数据集配置
        self.dataset_config = self._get_dataset_config()
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        
        # 初始化损失函数
        self.loss_metric = self._create_loss_metric()
        
        # 初始化日志
        self.logger = self._setup_logger()
        
        # 最佳指标跟踪
        self.best_img_auc = 0.0
        self.best_pix_auc = 0.0
    
    def _get_dataset_config(self):
        """获取数据集特定配置"""
        configs = {
            'btad': {
                'dataset_class': BTADDataset,
                'train_params': {
                    'root_dir': self.args.data_root,
                    'product': getattr(self.args, 'product', 'all'),
                    'mode': 'train',
                    'resize': self.args.resize,
                    'anomaly_source_dir': getattr(self.args, 'anomaly_source_dir', None)
                },
                'test_params': {
                    'root_dir': self.args.data_root,
                    'product': getattr(self.args, 'product', 'all'),
                    'mode': 'test',
                    'resize': self.args.resize
                },
                'log_prefix': 'btad',
                'model_prefix': f"btad_{getattr(self.args, 'product', 'all')}"
            },
            'dagm': {
                'dataset_class': DAGMDataset,
                'train_params': {
                    'root_dir': self.args.data_root,
                    'class_id': getattr(self.args, 'class_id', 'all'),
                    'mode': 'train',
                    'resize': self.args.resize
                },
                'test_params': {
                    'root_dir': self.args.data_root,
                    'class_id': getattr(self.args, 'class_id', 'all'),
                    'mode': 'test',
                    'resize': self.args.resize
                },
                'log_prefix': 'dagm',
                'model_prefix': f"dagm_{getattr(self.args, 'class_id', 'all')}"
            },
            'mpdd': {
                'dataset_class': MPDDDataset,
                'train_params': {
                    'root_dir': self.args.data_root,
                    'mode': 'train',
                    'resize': self.args.resize
                },
                'test_params': {
                    'root_dir': self.args.data_root,
                    'mode': 'test',
                    'resize': self.args.resize
                },
                'log_prefix': 'mpdd',
                'model_prefix': 'mpdd'
            },
            'mvtec': {
                'dataset_class': MVTecADDataset,
                'train_params': {
                    'root_dir': self.args.data_root,
                    'object_name': getattr(self.args, 'object_name', 'bottle'),
                    'mode': 'train',
                    'resize': self.args.resize,
                    'anomaly_source_dir': getattr(self.args, 'anomaly_source_dir', None)
                },
                'test_params': {
                    'root_dir': self.args.data_root,
                    'object_name': getattr(self.args, 'object_name', 'bottle'),
                    'mode': 'test',
                    'resize': self.args.resize
                },
                'log_prefix': 'mvtec',
                'model_prefix': f"mvtec_{getattr(self.args, 'object_name', 'bottle')}"
            },
            'visa': {
                'dataset_class': VisADataset,
                'train_params': {
                    'root_dir': self.args.data_root,
                    'mode': 'train',
                    'resize': self.args.resize
                },
                'test_params': {
                    'root_dir': self.args.data_root,
                    'mode': 'test',
                    'resize': self.args.resize
                },
                'log_prefix': 'visa',
                'model_prefix': 'visa'
            }
        }
        return configs.get(self.dataset_name, configs['btad'])
    
    def _create_model(self):
        """创建模型"""
        try:
            # 尝试使用统一模型
            model = create_model(self.dataset_name)
        except:
            # 回退到原始模型
            model = ResNet50_DRAEM()
        
        model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        return model
    
    def _create_loss_metric(self):
        """创建损失函数和评估指标"""
        try:
            return UnifiedLossMetric(self.dataset_name)
        except:
            # 回退到简单的损失函数类
            class SimpleLossMetric:
                def __init__(self):
                    self.focal_loss = FocalLoss()
                
                def compute_loss(self, recon_out, seg_out, img_gt, mask_gt):
                    try:
                        recon_loss = reconstruction_loss(recon_out, img_gt)
                        seg_loss = focal_loss(seg_out, mask_gt)
                    except:
                        # 简单的MSE和BCE损失
                        recon_loss = nn.MSELoss()(recon_out, img_gt)
                        seg_loss = self.focal_loss(torch.sigmoid(seg_out), mask_gt)
                    return recon_loss + seg_loss
            
            return SimpleLossMetric()
    
    def _setup_logger(self):
        """设置日志"""
        os.makedirs("logs", exist_ok=True)
        os.makedirs("trained_models", exist_ok=True)
        
        log_path = f"logs/{self.dataset_config['log_prefix']}_training.txt"
        metric_path = f"logs/{self.dataset_config['log_prefix']}_metrics.csv"
        
        log_file = open(log_path, 'w')
        metric_file = open(metric_path, 'w')
        metric_file.write("epoch,loss,img_auc,pix_auc\n")
        
        return {'log_file': log_file, 'metric_file': metric_file}
    
    def create_data_loaders(self):
        """创建数据加载器"""
        dataset_class = self.dataset_config['dataset_class']
        
        train_dataset = dataset_class(**self.dataset_config['train_params'])
        test_dataset = dataset_class(**self.dataset_config['test_params'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, test_loader
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch_data in train_loader:
            # 处理不同数据集的数据格式
            if len(batch_data) == 3:
                img_in, img_gt, mask_gt = batch_data
            else:
                # 某些数据集可能有不同的返回格式
                img_in, img_gt, mask_gt = batch_data[0], batch_data[1], batch_data[2]
            
            img_in = img_in.to(self.device)
            img_gt = img_gt.to(self.device)
            mask_gt = mask_gt.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 模型前向传播
            output = self.model(img_in)
            if isinstance(output, (tuple, list)):
                recon_out, seg_out = output
            else:
                recon_out, seg_out = img_in, output  # 某些模型只输出分割结果
            
            # 计算损失
            try:
                loss = self.loss_metric.compute_loss(recon_out, seg_out, img_gt, mask_gt)
            except:
                # 简单的损失计算
                recon_loss = nn.MSELoss()(recon_out, img_gt)
                seg_loss = nn.BCEWithLogitsLoss()(seg_out, mask_gt)
                loss = recon_loss + seg_loss
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * img_in.size(0)
        
        return running_loss / len(train_loader.dataset)
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        y_true_img, y_score_img = [], []
        y_true_pix, y_score_pix = [], []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 处理不同数据集的数据格式
                if len(batch_data) == 3:
                    imgs, labels, masks = batch_data
                else:
                    imgs, labels, masks = batch_data[0], batch_data[1], batch_data[2]
                
                imgs = imgs.to(self.device)
                
                # 模型预测
                output = self.model(imgs)
                if isinstance(output, (tuple, list)):
                    _, seg_logits = output
                else:
                    seg_logits = output
                
                # 计算概率图
                prob_map = torch.sigmoid(seg_logits)
                
                # 图像级分数
                score_img = float(prob_map.max().cpu().item())
                y_true_img.append(int(labels.item() if hasattr(labels, 'item') else labels[0]))
                y_score_img.append(score_img)
                
                # 像素级分数
                if masks is not None and masks[0] is not None:
                    gt_flat = masks.view(-1).cpu().numpy()
                    pr_flat = prob_map.view(-1).cpu().numpy()
                    y_true_pix.extend(gt_flat.tolist())
                    y_score_pix.extend(pr_flat.tolist())
        
        # 计算AUC
        img_auc = roc_auc_score(y_true_img, y_score_img) if len(set(y_true_img)) == 2 else 0.0
        pix_auc = roc_auc_score(y_true_pix, y_score_pix) if len(set(y_true_pix)) == 2 else 0.0
        
        return img_auc, pix_auc
    
    def save_model(self, epoch, img_auc, pix_auc):
        """保存模型"""
        state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        
        model_prefix = self.dataset_config['model_prefix']
        
        if img_auc > self.best_img_auc:
            self.best_img_auc = img_auc
            torch.save(state, f"trained_models/{model_prefix}_best_img_auc.pth")
        
        if pix_auc > self.best_pix_auc:
            self.best_pix_auc = pix_auc
            torch.save(state, f"trained_models/{model_prefix}_best_pix_auc.pth")
    
    def train(self):
        """主训练循环"""
        print(f"开始训练 {self.dataset_name.upper()} 数据集...")
        
        # 创建数据加载器
        train_loader, test_loader = self.create_data_loaders()
        
        # 训练循环
        for epoch in range(1, self.args.epochs + 1):
            # 训练一个epoch
            avg_loss = self.train_epoch(train_loader)
            
            # 评估
            img_auc, pix_auc = self.evaluate(test_loader)
            
            # 保存最佳模型
            self.save_model(epoch, img_auc, pix_auc)
            
            # 记录日志
            log_msg = f"Epoch {epoch}/{self.args.epochs} - Loss: {avg_loss:.4f}, Img_AUC: {img_auc:.4f}, Pix_AUC: {pix_auc:.4f}"
            print(log_msg)
            
            self.logger['log_file'].write(log_msg + "\n")
            self.logger['metric_file'].write(f"{epoch},{avg_loss:.6f},{img_auc:.4f},{pix_auc:.4f}\n")
        
        # 关闭日志文件
        self.logger['log_file'].close()
        self.logger['metric_file'].close()
        
        print(f"训练完成！")
        print(f"最佳 Img_AUC: {self.best_img_auc:.4f}")
        print(f"最佳 Pix_AUC: {self.best_pix_auc:.4f}")


class FocalLoss(nn.Module):
    """Focal Loss实现"""
    def __init__(self, alpha=0.7, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        eps = 1e-6
        p = pred.clamp(eps, 1-eps)
        pt = torch.where(target == 1, p, 1-p)
        w = torch.where(target == 1, (1-p)**self.gamma, p**self.gamma)
        a = torch.where(target == 1, self.alpha, 1-self.alpha)
        return (-a * w * torch.log(pt)).mean()


def create_trainer(dataset_name, args):
    """训练器工厂函数"""
    return UnifiedTrainer(dataset_name, args)


def get_dataset_info(dataset_name):
    """获取数据集信息"""
    info = {
        'btad': {
            'name': 'BTAD',
            'description': 'Beantech Anomaly Detection Dataset',
            'required_args': ['data_root', 'product']
        },
        'dagm': {
            'name': 'DAGM',
            'description': 'DAGM 2007 Dataset',
            'required_args': ['data_root', 'class_id']
        },
        'mpdd': {
            'name': 'MPDD',
            'description': 'Metal Parts Defect Dataset',
            'required_args': ['data_root']
        },
        'mvtec': {
            'name': 'MVTec AD',
            'description': 'MVTec Anomaly Detection Dataset',
            'required_args': ['data_root', 'object_name']
        },
        'visa': {
            'name': 'VisA',
            'description': 'Visual Anomaly Dataset',
            'required_args': ['data_root']
        }
    }
    return info.get(dataset_name.lower(), {})


def main():
    parser = argparse.ArgumentParser(description="统一训练脚本 - 支持多个异常检测数据集")
    
    # 基本参数
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['btad', 'dagm', 'mpdd', 'mvtec', 'visa'],
                        help='数据集名称')
    parser.add_argument('--data_root', type=str, required=False,
                        help='数据集根目录')
    parser.add_argument('--resize', type=int, default=256,
                        help='输入图像尺寸')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    
    # 数据集特定参数
    parser.add_argument('--product', type=str, default='all',
                        help='BTAD数据集产品编号 (01/02/03/all)')
    parser.add_argument('--class_id', type=str, default='all',
                        help='DAGM数据集类别编号 (1-10/all)')
    parser.add_argument('--object_name', type=str, default='bottle',
                        help='MVTec数据集对象名称')
    parser.add_argument('--anomaly_source_dir', type=str, default=None,
                        help='伪缺陷纹理目录')
    
    # 功能参数
    parser.add_argument('--info', action='store_true',
                        help='显示数据集信息')
    parser.add_argument('--test', action='store_true',
                        help='测试模式（快速验证）')
    
    args = parser.parse_args()
    
    # 显示数据集信息
    if args.info:
        info = get_dataset_info(args.dataset)
        print(f"数据集: {info.get('name', args.dataset)}")
        print(f"描述: {info.get('description', 'N/A')}")
        print(f"必需参数: {', '.join(info.get('required_args', []))}")
        return
    
    # 测试模式
    if args.test:
        args.epochs = 2
        args.batch_size = 2
        print(f"测试模式：epochs={args.epochs}, batch_size={args.batch_size}")
    
    # 创建并运行训练器
    trainer = create_trainer(args.dataset, args)
    trainer.train()


if __name__ == "__main__":
    main()