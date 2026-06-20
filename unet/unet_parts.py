"""U-Net 使用的基础网络组件。

本文件只定义可复用的小模块，完整网络结构在 ``unet_model.py`` 中组装。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """连续执行两次 ``卷积 → 批归一化 → ReLU``。

    3×3 卷积使用 padding=1，因此不会改变特征图的宽和高。
    BatchNorm 可稳定各层输入分布，ReLU 则提供非线性表达能力。
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # 默认让中间通道数与输出通道数一致；Up 模块也可显式指定较小的中间通道数。
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # bias=False 是因为紧随其后的 BatchNorm 已包含可学习的平移参数。
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """保持空间尺寸不变，只将通道数转换为 out_channels。"""

        return self.double_conv(x)


class Down(nn.Module):
    """编码器中的一次下采样：最大池化后接 DoubleConv。

    2×2 最大池化令宽、高各减半；卷积块负责提取更高层语义特征。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """输入 [N, C, H, W]，输出空间尺寸约为 [N, C', H/2, W/2]。"""

        return self.maxpool_conv(x)


class Up(nn.Module):
    """解码器中的一次上采样、跳跃连接和卷积融合。

    参数:
        in_channels: 拼接后送入卷积块的总通道数。
        out_channels: 该解码层最终输出的通道数。
        bilinear: 是否使用双线性插值；否则使用可学习的转置卷积。
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 双线性插值没有可学习参数，只放大空间尺寸；后续 DoubleConv 负责通道压缩。
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 转置卷积同时把宽高放大 2 倍，并把低分辨率分支的通道数减半。
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """融合解码器特征 ``x1`` 与编码器跳跃连接特征 ``x2``。

        ``x1`` 来自更深一层，分辨率较低；``x2`` 来自编码器，包含同尺度的
        高分辨率细节。二者按通道维拼接后由 DoubleConv 完成特征融合。
        """

        # 先把深层特征的宽高扩大 2 倍。
        x1 = self.up(x1)

        # 输入图像尺寸不一定能被 16 整除，连续下采样再上采样后可能出现 1 像素误差。
        # 这里计算编码器特征与上采样结果在高、宽方向的尺寸差。
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 将差值平均补到 x1 两侧，使 x1 与 x2 的空间尺寸完全一致。
        # F.pad 的顺序是 [左, 右, 上, 下]。
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 若遇到奇数尺寸导致的 padding 问题，可参考下列历史修复：
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        # dim=1 表示按通道维拼接，空间尺寸保持不变。
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """使用 1×1 卷积把特征通道映射为类别通道。"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """返回每个类别的逐像素 logits，不改变特征图宽高。"""

        return self.conv(x)
