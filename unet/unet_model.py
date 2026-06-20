"""将 U-Net 的基础组件组装成完整网络。

U-Net 由两部分组成：

- 编码器（下采样路径）：逐级减小空间尺寸、增加通道数，用于提取语义特征；
- 解码器（上采样路径）：逐级恢复空间尺寸，并融合编码器中相同尺度的细节特征。

这种对称结构和跨层跳跃连接使模型既能理解“图中是什么”，又能较准确地恢复
目标边界，因此非常适合医学图像、道路、车辆等像素级分割任务。
"""

# unet_parts 中定义了 DoubleConv、Down、Up 和 OutConv 等基础模块。
from .unet_parts import *


class UNet(nn.Module):
    """标准二维 U-Net。

    参数:
        n_channels: 输入图像通道数，RGB 图像为 3，灰度图像为 1。
        n_classes: 输出类别数。多分类通常等于类别总数；二值任务也可设置为 1。
        bilinear: 为 True 时使用双线性插值上采样，否则使用转置卷积。
    """

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        # 保存这些配置，训练与预测代码会读取它们来检查输入和选择损失计算方式。
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器：每经过一个 Down，特征图宽高减半，通道数通常翻倍。
        # 假设输入为 [N, 3, H, W]，下面各层输出依次约为：
        # x1: [N, 64, H, W]
        # x2: [N, 128, H/2, W/2]
        # x3: [N, 256, H/4, W/4]
        # x4: [N, 512, H/8, W/8]
        # x5: [N, 1024/factor, H/16, W/16]
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))

        # 双线性插值本身不改变通道数，因此需要把网络底部通道数减半，
        # 以保持模型规模与转置卷积版本大致一致。
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # 解码器：Up 会先放大低分辨率特征，再与编码器同尺度特征拼接。
        # 跳跃连接把浅层的边缘、纹理和位置信息直接传递到解码器。
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        # 1×1 卷积只变换通道数，不改变空间尺寸。
        # 最终每个输出通道对应一个类别的像素级 logits。
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        """执行一次前向传播并返回分割 logits。

        输入:
            x: 形状为 ``[batch, n_channels, height, width]`` 的图像张量。

        输出:
            形状为 ``[batch, n_classes, height, width]`` 的 logits。
            这里不执行 softmax/sigmoid，因为损失函数通常会在内部以更稳定的方式处理。
        """

        # 保存每一级编码器特征，供解码阶段的跳跃连接使用。
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 每一级 Up 的第二个输入都是编码器中相同空间尺度的特征。
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 将 64 通道特征映射成 n_classes 个类别通道。
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """为主要模块启用梯度检查点以降低训练显存占用。

        梯度检查点不会在前向传播时保存所有中间激活，而是在反向传播时重新计算，
        因而以额外计算时间换取更低的显存使用量。该方法通常在捕获到 CUDA
        显存不足后调用。
        """

        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
