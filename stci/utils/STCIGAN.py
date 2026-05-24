import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ 

# 引入 Swin Transformer 相关的层，假设你已经有这些导入
# from models.swin_transformer_3d import SwinTransformer3D # 举例，你的路径可能不同
# from models.patch_embed import PatchEmbed3D # 假设你有3D Patch Embedding

class Discriminator(nn.Module):
    def __init__(self,
                 in_channels=3,                     # 输入视频的通道数 (e.g., RGB=3)
                 base_dim=64,                       # 判别器起始特征维度
                 num_layers=4,                      # 判别器的下采样层数
                 norm_layer=nn.InstanceNorm3d,      # 推荐使用 InstanceNorm3d 或 SpectralNorm
                 cond_channels=None,                # 用于接收模板条件的通道数
                 img_size=(16, 256, 256)            # 期望判别器处理的视频尺寸 (T, H, W)
                ):
        super().__init__()

        self.main_stream = nn.Sequential()

        # 初始卷积层：将视频输入转换为特征
        self.main_stream.add_module(
            'initial_conv',
            nn.Conv3d(in_channels, base_dim, kernel_size=3, stride=1, padding=1) # 4x4x4 kernel, stride 2, reduces T/H/W by half
        )
        self.main_stream.add_module('initial_act', nn.LeakyReLU(0.2, inplace=True))

        current_dim = base_dim
        # 下采样层：逐步提取特征并降低分辨率
        for i in range(1, num_layers):
            next_dim = min(current_dim * 2, 512) 
            self.main_stream.add_module(
                f'down_block_{i}',
                nn.Sequential(
                    nn.Conv3d(current_dim, next_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
                    norm_layer(next_dim),                                                   # 对判别器，推荐InstanceNorm3d或SpectralNorm
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            current_dim = next_dim

        # 最终卷积层：输出一个特征图，代表判别结果
        # 判别器通常不进行全局平均池化，而是输出一个 "PatchGAN" 风格的特征图
        self.output_conv = nn.Conv3d(current_dim, 1, kernel_size=3, stride=1, padding=1) # 输出1通道的logit图

        # 模板条件化层（可选，用于条件判别器）
        self.cond_net = None
        if cond_channels is not None:
            # 模板条件网络：将模板特征编码，并融合到判别器中
            # 可以是简单的MLP，也可以是匹配判别器下采样结构的小型CNN
            self.cond_net = nn.Sequential(
                nn.Conv3d(cond_channels, base_dim, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
                # ... 更多层以匹配判别器的下采样
            )
            # 融合层：将模板条件特征融合到判别器主干的某个中间层
            self.fusion_conv = nn.Conv3d(base_dim * 2, base_dim * 2, kernel_size=1) # 示例融合点

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv3d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, template_cond=None):
        # x: (B, C, T, H, W) 真实视频或生成视频
        # 主干网络处理
        main_features = self.main_stream(x)                                     # 例如，经过3层下采样后，T/H/W 缩小 8 倍

        # 融合模板条件
        if self.cond_net is not None and template_cond is not None:
            cond_features = self.cond_net(template_cond)
           
            if cond_features.shape[2:] != main_features.shape[2:]:               # 可能需要resize cond_features 以匹配 main_features 的尺寸
                 
                cond_features = torch.nn.functional.interpolate(                 # 使用3D插值进行尺寸匹配
                    cond_features,
                    size=main_features.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
            fused_features = main_features + self.fusion_conv(cond_features)     # 简单相加融合
            output = self.output_conv(fused_features)
        else:
            output = self.output_conv(main_features)

        # 判别器输出的是 logits，后续会用 BCEWithLogitsLoss
        return output

class TemplateEncoder(nn.Module):
    def __init__(self, in_channels, base_dim=32, num_down_blocks=3):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        current_dim = in_channels
        self.encoder_blocks.append(
            nn.Sequential(
                nn.Conv3d(current_dim, base_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        current_dim = base_dim
        for i in range(num_down_blocks):
            next_dim = current_dim * 2
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv3d(current_dim, next_dim, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm3d(next_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            current_dim = next_dim
        self.final_conv = nn.Conv3d(current_dim, current_dim, kernel_size=3, stride=1, padding=1)
        self.cond_vec_mlp = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                                # 全局平均池化为 (B, C, 1, 1, 1)
            nn.Flatten(),
            nn.Linear(current_dim, 256)                             # 映射到固定维度，例如 256
        )

    def forward(self, template):
        x = template
        for block in self.encoder_blocks:
            x = block(x)
        final_feature_map = self.final_conv(x)        
        cond_vec = self.cond_vec_mlp(final_feature_map)             # 为 Generator 提供一个扁平化的条件向量
        return cond_vec, final_feature_map                          # 输出一个关于模板的map，一个关于模板的vector
