import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------
#                           ***RDN***
# --------------------------------------------------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class Block_RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(Block_RDB, self).__init__()
        # 传入一个可迭代对象(列表、元组)的模块，并使用*进行解包/确保sequential解析到的是DenseLayer中的模块，而不是一个列表
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class Block_RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(Block_RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([Block_RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(Block_RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, 1, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x
    
    
# --------------------------------------------------------------------
#                    ***Basic Module for Birnat***
# --------------------------------------------------------------------
class self_attention(nn.Module):
    def __init__(self, ch):
        super(self_attention, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch // 8, 1)
        self.conv2 = nn.Conv2d(ch, ch // 8, 1)
        self.conv3 = nn.Conv2d(ch, ch, 1)
        self.conv4 = nn.Conv2d(ch, ch, 1)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.ch = ch

    def forward(self, x):
        batch_size = x.shape[0]

        f = self.conv1(x)
        g = self.conv2(x)
        h = self.conv3(x)
        ht = h.reshape([batch_size, self.ch, -1])

        ft = f.reshape([batch_size, self.ch // 8, -1])
        n = torch.matmul(ft.permute([0, 2, 1]), g.reshape([batch_size, self.ch // 8, -1]))
        beta = F.softmax(n, dim=-1)

        o = torch.matmul(ht, beta)
        o = o.reshape(x.shape)  # [bs, C, h, w]

        o = self.conv4(o)

        x = self.gamma1 * o + x

        return x


class res_part(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(res_part, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = x1 + x
        x1 = self.conv2(x)
        x = x1 + x
        x1 = self.conv3(x)
        x = x1 + x
        return x


class down_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 40, 3, stride=1, padding=1),
            nn.Conv2d(40, 30, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(30, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.Conv2d(20, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x