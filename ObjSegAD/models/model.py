# model.py
# 统一的模型架构模块，支持多个数据集

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple, Optional, Union

# ============================================================================
# 注意力机制模块
# ============================================================================

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力：全局平均 & 最大池化 -> FC -> Sigmoid
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        # 空间注意力：按通道求平均 & 最大 -> concat -> conv -> sigmoid
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(feat))
        return x * scale


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================================
# 解码器模块
# ============================================================================

class DecoderBlock(nn.Module):
    """解码器块，包含上采样、特征融合和CBAM注意力"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, 
                 use_cbam: bool = True, style: str = 'detailed'):
        super().__init__()
        self.style = style
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        if style == 'detailed':
            # BTAD/DAGM风格：详细的层定义
            self.conv = nn.Conv2d(out_channels + skip_channels, out_channels, 
                                kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()
        else:
            # MPDD/MVTEC/VISA风格：紧凑的Sequential定义
            layers = [
                nn.Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1),
                nn.ReLU(True)
            ]
            if use_cbam:
                layers.append(CBAM(out_channels))
            self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 上采样
        up = self.up(x)
        # 调整尺寸匹配
        up = F.interpolate(up, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        # 特征融合
        concat = torch.cat([up, skip], dim=1)
        
        if self.style == 'detailed':
            out = self.conv(concat)
            out = self.relu(out)
            out = self.cbam(out)
        else:
            out = self.decoder(concat)
        
        return out


# ============================================================================
# 统一的ResNet50-DRAEM模型
# ============================================================================

class UnifiedResNet50DRAEM(nn.Module):
    """
    统一的ResNet50-DRAEM模型，支持不同数据集的配置
    基于ResNet50编码器，双解码器（重构+分割）结构，集成CBAM注意力
    """
    
    def __init__(self, dataset: str = 'general', pretrained: bool = True):
        super().__init__()
        self.dataset = dataset
        self.config = self._get_dataset_config(dataset)
        
        # 初始化ResNet50编码器
        resnet = models.resnet50(pretrained=pretrained)
        self._build_encoder(resnet)
        self._build_decoders()
    
    def _get_dataset_config(self, dataset: str) -> Dict:
        """获取数据集特定配置"""
        configs = {
            'btad': {
                'style': 'detailed',
                'use_cbam': True,
                'reduction': 16,
                'kernel_size': 7
            },
            'dagm': {
                'style': 'detailed',
                'use_cbam': True,
                'reduction': 16,
                'kernel_size': 7
            },
            'mpdd': {
                'style': 'compact',
                'use_cbam': True,
                'reduction': 16,
                'kernel_size': 7
            },
            'mvtec': {
                'style': 'compact',
                'use_cbam': True,
                'reduction': 16,
                'kernel_size': 7
            },
            'visa': {
                'style': 'compact',
                'use_cbam': True,
                'reduction': 16,
                'kernel_size': 7
            }
        }
        return configs.get(dataset, configs['btad'])
    
    def _build_encoder(self, resnet: nn.Module):
        """构建编码器"""
        self.encoder0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # 输出[B,64,H/4,W/4]
        self.encoder1 = resnet.layer1  # [B,256,H/4,W/4]
        self.encoder2 = resnet.layer2  # [B,512,H/8,W/8]
        self.encoder3 = resnet.layer3  # [B,1024,H/16,W/16]
        self.encoder4 = resnet.layer4  # [B,2048,H/32,W/32]
    
    def _build_decoders(self):
        """构建解码器"""
        style = self.config['style']
        use_cbam = self.config['use_cbam']
        
        # 重构分支解码器
        self.dec4_rec = DecoderBlock(2048, 1024, 1024, use_cbam, style)
        self.dec3_rec = DecoderBlock(1024, 512, 512, use_cbam, style)
        self.dec2_rec = DecoderBlock(512, 256, 256, use_cbam, style)
        self.dec1_rec = DecoderBlock(256, 64, 64, use_cbam, style)
        self.final_rec = nn.Conv2d(64, 3, kernel_size=1)
        
        # 分割分支解码器
        self.dec4_seg = DecoderBlock(2048, 1024, 1024, use_cbam, style)
        self.dec3_seg = DecoderBlock(1024, 512, 512, use_cbam, style)
        self.dec2_seg = DecoderBlock(512, 256, 256, use_cbam, style)
        self.dec1_seg = DecoderBlock(256, 64, 64, use_cbam, style)
        self.final_seg = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            Tuple[重构图像, 分割logits]
        """
        # 编码器
        e0 = self.encoder0(x)  # [B,64,H/4,W/4]
        e1 = self.encoder1(e0) # [B,256,H/4,W/4]
        e2 = self.encoder2(e1) # [B,512,H/8,W/8]
        e3 = self.encoder3(e2) # [B,1024,H/16,W/16]
        e4 = self.encoder4(e3) # [B,2048,H/32,W/32]
        
        # 重构分支解码器
        d4r = self.dec4_rec(e4, e3)
        d3r = self.dec3_rec(d4r, e2)
        d2r = self.dec2_rec(d3r, e1)
        d1r = self.dec1_rec(d2r, e0)
        
        # 输出重构图像
        out_rec = F.interpolate(d1r, size=x.shape[-2:], mode='bilinear', align_corners=False)
        out_rec = self.final_rec(out_rec)
        
        # 分割分支解码器
        s4 = self.dec4_seg(e4, e3)
        s3 = self.dec3_seg(s4, e2)
        s2 = self.dec2_seg(s3, e1)
        s1 = self.dec1_seg(s2, e0)
        
        # 输出分割logits
        out_seg = F.interpolate(s1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        out_seg = self.final_seg(out_seg)
        
        return out_rec, out_seg
    
    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """获取异常概率图"""
        _, seg_logits = self.forward(x)
        return torch.sigmoid(seg_logits)
    
    def get_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """获取重构图像"""
        rec_img, _ = self.forward(x)
        return rec_img


# ============================================================================
# 模型工厂函数
# ============================================================================

def create_model(dataset: str = 'general', pretrained: bool = True, **kwargs) -> UnifiedResNet50DRAEM:
    """
    创建指定数据集的模型
    
    Args:
        dataset: 数据集名称 ('btad', 'dagm', 'mpdd', 'mvtec', 'visa', 'general')
        pretrained: 是否使用预训练权重
        **kwargs: 其他参数
    
    Returns:
        配置好的模型实例
    """
    return UnifiedResNet50DRAEM(dataset=dataset, pretrained=pretrained)


def get_model_info(dataset: str = 'general') -> Dict:
    """
    获取模型配置信息
    
    Args:
        dataset: 数据集名称
    
    Returns:
        模型配置字典
    """
    model = UnifiedResNet50DRAEM(dataset=dataset, pretrained=False)
    return {
        'dataset': dataset,
        'config': model.config,
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


# ============================================================================
# 向后兼容类（保持原有接口）
# ============================================================================

class ResNet50_DRAEM(UnifiedResNet50DRAEM):
    """向后兼容的ResNet50_DRAEM类"""
    
    def __init__(self, dataset: str = 'general'):
        super().__init__(dataset=dataset, pretrained=True)


# ============================================================================
# 主函数和命令行接口
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='统一模型架构模块')
    parser.add_argument('--dataset', type=str, default='general',
                       choices=['btad', 'dagm', 'mpdd', 'mvtec', 'visa', 'general'],
                       help='数据集名称')
    parser.add_argument('--test', action='store_true', help='运行测试')
    parser.add_argument('--info', action='store_true', help='显示模型信息')
    parser.add_argument('--batch-size', type=int, default=2, help='测试批次大小')
    parser.add_argument('--input-size', type=int, default=256, help='输入图像尺寸')
    
    args = parser.parse_args()
    
    if args.info:
        print(f"模型信息 - 数据集: {args.dataset}")
        info = get_model_info(args.dataset)
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    if args.test:
        print(f"测试数据集 {args.dataset} 的模型...")
        
        # 创建模型
        model = create_model(args.dataset, pretrained=False)
        model.eval()
        
        # 测试输入
        batch_size = args.batch_size
        input_size = args.input_size
        test_input = torch.randn(batch_size, 3, input_size, input_size)
        
        print(f"输入尺寸: {test_input.shape}")
        
        # 前向传播测试
        with torch.no_grad():
            rec_output, seg_output = model(test_input)
            anomaly_map = model.get_anomaly_map(test_input)
            reconstruction = model.get_reconstruction(test_input)
        
        print("模型输出测试结果:")
        print(f"  重构输出尺寸: {rec_output.shape}")
        print(f"  分割输出尺寸: {seg_output.shape}")
        print(f"  异常概率图尺寸: {anomaly_map.shape}")
        print(f"  重构图像尺寸: {reconstruction.shape}")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数数: {trainable_params:,}")
        
        # 测试向后兼容性
        print("\n测试向后兼容性...")
        legacy_model = ResNet50_DRAEM(args.dataset)
        legacy_model.eval()
        
        with torch.no_grad():
            legacy_rec, legacy_seg = legacy_model(test_input)
        
        print(f"  兼容模型重构输出尺寸: {legacy_rec.shape}")
        print(f"  兼容模型分割输出尺寸: {legacy_seg.shape}")
        
        print(f"\n{args.dataset} 数据集的模型测试完成！")
    
    if not args.test and not args.info:
        print("统一模型架构模块已加载")
        print("支持的数据集:", ['btad', 'dagm', 'mpdd', 'mvtec', 'visa'])
        print("使用 --test 参数运行测试")
        print("使用 --info 参数查看模型信息")