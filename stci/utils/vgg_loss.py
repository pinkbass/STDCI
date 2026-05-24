"""
VGG中间层感知损失模块
使用VGG网络的中间层特征进行感知损失计算，相比最终输出能提供更好的特征表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    基于VGG中间层的感知损失
    使用VGG网络的多个中间层特征进行损失计算
    """
    
    def __init__(self, feature_layers=None, weights=None, normalize=True):
        """
        Args:
            feature_layers: 要使用的VGG层索引列表，默认为[3, 8, 15, 22]
            weights: 各层损失的权重，默认为[1.0, 1.0, 1.0, 1.0]
            normalize: 是否对特征进行L2归一化
        """
        super(VGGPerceptualLoss, self).__init__()
        
        # 默认使用VGG16的conv1_2, conv2_2, conv3_3, conv4_3层
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]  # VGG16的中间层
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]
            
        self.feature_layers = feature_layers
        self.weights = weights
        self.normalize = normalize
        
        # 加载预训练的VGG16模型
        vgg = models.vgg16(pretrained=True)
        vgg.eval()
        
        # 提取特征层
        self.features = nn.ModuleList()
        for i in range(max(feature_layers) + 1):
            self.features.append(vgg.features[i])
        
        # 冻结VGG参数
        for param in self.parameters():
            param.requires_grad = False
            
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
        pred_features = []
        target_features = []
        
        # 逐层提取特征
        for i, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            
            if i in self.feature_layers:
                pred_features.append(pred)
                target_features.append(target)
        
        # 计算各层损失
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            if self.normalize:
                # L2归一化
                pred_feat = F.normalize(pred_feat, p=2, dim=1)
                target_feat = F.normalize(target_feat, p=2, dim=1)
            
            # 计算L2损失
            layer_loss = F.mse_loss(pred_feat, target_feat)
            total_loss += self.weights[i] * layer_loss
            
        return total_loss


class VGGStyleLoss(nn.Module):
    """
    基于VGG中间层的风格损失
    使用Gram矩阵计算风格相似性
    """
    
    def __init__(self, feature_layers=None, weights=None):
        """
        Args:
            feature_layers: 要使用的VGG层索引列表
            weights: 各层损失的权重
        """
        super(VGGStyleLoss, self).__init__()
        
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]
            
        self.feature_layers = feature_layers
        self.weights = weights
        
        # 加载VGG16
        vgg = models.vgg16(pretrained=True)
        vgg.eval()
        
        self.features = nn.ModuleList()
        for i in range(max(feature_layers) + 1):
            self.features.append(vgg.features[i])
            
        for param in self.parameters():
            param.requires_grad = False
            
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
        
        for i, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            
            if i in self.feature_layers:
                pred_gram = self.gram_matrix(pred)
                target_gram = self.gram_matrix(target)
                
                layer_loss = F.mse_loss(pred_gram, target_gram)
                total_loss += self.weights[layer_idx] * layer_loss
                layer_idx += 1
                
        return total_loss


class CombinedVGGLoss(nn.Module):
    """
    结合感知损失和风格损失的VGG损失
    """
    
    def __init__(self, perceptual_weight=1.0, style_weight=0.1, 
                 feature_layers=None, perceptual_weights=None, style_weights=None):
        """
        Args:
            perceptual_weight: 感知损失权重
            style_weight: 风格损失权重
            feature_layers: 特征层索引
            perceptual_weights: 感知损失各层权重
            style_weights: 风格损失各层权重
        """
        super(CombinedVGGLoss, self).__init__()
        
        self.perceptual_loss = VGGPerceptualLoss(feature_layers, perceptual_weights)
        self.style_loss = VGGStyleLoss(feature_layers, style_weights)
        
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        
    def forward(self, pred, target):
        """
        计算组合损失
        Args:
            pred: 预测图像
            target: 目标图像
        """
        perceptual = self.perceptual_loss(pred, target)
        style = self.style_loss(pred, target)
        
        total_loss = (self.perceptual_weight * perceptual + 
                     self.style_weight * style)
        
        return total_loss, perceptual, style


def create_vgg_loss(loss_type='perceptual', **kwargs):
    """
    创建VGG损失的便捷函数
    Args:
        loss_type: 损失类型 ('perceptual', 'style', 'combined')
        **kwargs: 其他参数
    """
    if loss_type == 'perceptual':
        return VGGPerceptualLoss(**kwargs)
    elif loss_type == 'style':
        return VGGStyleLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedVGGLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# 测试函数
if __name__ == "__main__":
    # 测试VGG损失模块
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    pred = torch.randn(2, 1, 64, 64).to(device)
    target = torch.randn(2, 1, 64, 64).to(device)
    
    # 测试感知损失
    print("测试VGG感知损失...")
    perceptual_loss = VGGPerceptualLoss().to(device)
    loss = perceptual_loss(pred, target)
    print(f"感知损失: {loss.item():.6f}")
    
    # 测试风格损失
    print("测试VGG风格损失...")
    style_loss = VGGStyleLoss().to(device)
    loss = style_loss(pred, target)
    print(f"风格损失: {loss.item():.6f}")
    
    # 测试组合损失
    print("测试组合VGG损失...")
    combined_loss = CombinedVGGLoss().to(device)
    total, perc, style = combined_loss(pred, target)
    print(f"总损失: {total.item():.6f}, 感知: {perc.item():.6f}, 风格: {style.item():.6f}")
    
    print("VGG损失模块测试完成!")
