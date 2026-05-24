"""
简化的VGG中间层损失实现
不依赖torchvision，直接定义VGG网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVGGLoss(nn.Module):
    """
    简化的VGG中间层感知损失
    直接定义VGG16结构，使用指定的中间层
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22], weights=None, normalize=True):
        """
        Args:
            feature_layers: 要使用的VGG层索引列表
            weights: 各层损失的权重
            normalize: 是否对特征进行L2归一化
        """
        super(SimpleVGGLoss, self).__init__()
        
        self.feature_layers = feature_layers
        self.weights = weights or [1.0] * len(feature_layers)
        self.normalize = normalize
        
        # 定义VGG16特征提取器
        self.features = nn.ModuleList([
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),      # 0
            nn.ReLU(inplace=True),               # 1
            nn.Conv2d(64, 64, 3, padding=1),     # 2
            nn.ReLU(inplace=True),               # 3  <- conv1_2
            nn.MaxPool2d(2, 2),                  # 4
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),    # 5
            nn.ReLU(inplace=True),               # 6
            nn.Conv2d(128, 128, 3, padding=1),   # 7
            nn.ReLU(inplace=True),               # 8  <- conv2_2
            nn.MaxPool2d(2, 2),                  # 9
            
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),   # 10
            nn.ReLU(inplace=True),               # 11
            nn.Conv2d(256, 256, 3, padding=1),   # 12
            nn.ReLU(inplace=True),               # 13
            nn.Conv2d(256, 256, 3, padding=1),   # 14
            nn.ReLU(inplace=True),               # 15 <- conv3_3
            nn.MaxPool2d(2, 2),                  # 16
            
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),   # 17
            nn.ReLU(inplace=True),               # 18
            nn.Conv2d(512, 512, 3, padding=1),   # 19
            nn.ReLU(inplace=True),               # 20
            nn.Conv2d(512, 512, 3, padding=1),   # 21
            nn.ReLU(inplace=True),               # 22 <- conv4_3
            nn.MaxPool2d(2, 2),                  # 23
            
            # conv5
            nn.Conv2d(512, 512, 3, padding=1),   # 24
            nn.ReLU(inplace=True),               # 25
            nn.Conv2d(512, 512, 3, padding=1),   # 26
            nn.ReLU(inplace=True),               # 27
            nn.Conv2d(512, 512, 3, padding=1),   # 28
            nn.ReLU(inplace=True),               # 29 <- conv5_3
        ])
        
        # 加载预训练权重（简化版本，使用随机初始化）
        self._initialize_weights()
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
            
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, pred, target):
        """
        计算感知损失
        Args:
            pred: 预测图像，形状为 (B, C, H, W)
            target: 目标图像，形状为 (B, C, H, W)
        Returns:
            loss: 感知损失值
        """
        # 确保输入是3通道
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
            
        # 归一化到ImageNet标准
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        total_loss = 0.0
        layer_idx = 0
        
        # 逐层提取特征
        for i, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            
            if i in self.feature_layers:
                if self.normalize:
                    # L2归一化
                    pred_feat = F.normalize(pred, p=2, dim=1)
                    target_feat = F.normalize(target, p=2, dim=1)
                else:
                    pred_feat = pred
                    target_feat = target
                
                # 计算L2损失
                layer_loss = F.mse_loss(pred_feat, target_feat)
                total_loss += self.weights[layer_idx] * layer_loss
                layer_idx += 1
        
        return total_loss


class VGGStyleLoss(nn.Module):
    """
    基于VGG中间层的风格损失
    使用Gram矩阵计算风格相似性
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22], weights=None):
        super(VGGStyleLoss, self).__init__()
        
        self.feature_layers = feature_layers
        self.weights = weights or [1.0] * len(feature_layers)
        
        # 使用相同的VGG结构
        self.vgg_loss = SimpleVGGLoss(feature_layers, [1.0] * len(feature_layers), normalize=False)
        
    def gram_matrix(self, x):
        """计算Gram矩阵"""
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        gram = torch.bmm(x, x.transpose(1, 2))
        return gram / (c * h * w)
        
    def forward(self, pred, target):
        """
        计算风格损失
        Args:
            pred: 预测图像
            target: 目标图像
        """
        # 确保3通道
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
            
        # 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        total_loss = 0.0
        layer_idx = 0
        
        # 逐层提取特征并计算Gram矩阵
        for i, layer in enumerate(self.vgg_loss.features):
            pred = layer(pred)
            target = layer(target)
            
            if i in self.feature_layers:
                pred_gram = self.gram_matrix(pred)
                target_gram = self.gram_matrix(target)
                
                layer_loss = F.mse_loss(pred_gram, target_gram)
                total_loss += self.weights[layer_idx] * layer_loss
                layer_idx += 1
                
        return total_loss


def create_vgg_loss(loss_type='perceptual', **kwargs):
    """
    创建VGG损失的便捷函数
    Args:
        loss_type: 损失类型 ('perceptual', 'style')
        **kwargs: 其他参数
    """
    if loss_type == 'perceptual':
        return SimpleVGGLoss(**kwargs)
    elif loss_type == 'style':
        return VGGStyleLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# 测试函数
if __name__ == "__main__":
    print("简化VGG损失模块测试")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    pred = torch.randn(2, 1, 64, 64).to(device)
    target = torch.randn(2, 1, 64, 64).to(device)
    
    print(f"输入形状: {pred.shape}")
    
    # 测试不同层配置
    configs = [
        ([3], [1.0], "conv1_2"),
        ([8], [1.0], "conv2_2"),
        ([15], [1.0], "conv3_3"),
        ([22], [1.0], "conv4_3"),
        ([3, 8, 15, 22], [1.0, 1.0, 1.0, 1.0], "所有层"),
        ([3, 8, 15, 22], [0.1, 0.1, 1.0, 1.0], "深层权重更高"),
    ]
    
    for layers, weights, name in configs:
        try:
            vgg_loss = SimpleVGGLoss(layers, weights).to(device)
            loss = vgg_loss(pred, target)
            print(f"{name:15s}: 损失 = {loss.item():.6f}")
        except Exception as e:
            print(f"{name:15s}: 错误 - {e}")
    
    print("\n测试完成!")








