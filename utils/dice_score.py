"""Dice 相似系数及 Dice 损失。

Dice 用于衡量两个区域的重合程度：

    Dice = 2 × |预测 ∩ 真实| / (|预测| + |真实|)

结果范围通常为 0～1，1 表示完全重合。与像素准确率相比，Dice 对前景区域较小、
类别不平衡的分割任务更有参考价值。
"""

import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """计算二值掩码的平均 Dice 系数。

    参数:
        input: 预测掩码或前景概率。
        target: 与 input 同形状的真实掩码。
        reduce_batch_first: 是否把批次维与空间维一起求和。
        epsilon: 防止分母为零的平滑项。
    """

    # 预测与标签必须逐像素对应。
    assert input.size() == target.size()
    # 对二维单张掩码不存在批次维，因此不允许要求合并批次。
    assert input.dim() == 3 or not reduce_batch_first

    # 普通模式分别对最后两个空间维求和；合并批次时同时对后三维求和。
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # inter 是公式中的两倍交集；sets_sum 是预测区域与真实区域面积之和。
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    # 当预测和标签都为空时，将分母替换成交集值，使该样本经平滑后得到 Dice=1。
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    # 对批次或通道产生的多个 Dice 值取平均。
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """计算多分类掩码的平均 Dice 系数。

    输入通常是 ``[N, C, H, W]`` 的 one-hot 张量。先把批次维和类别维合并，
    再复用二值 Dice 计算，可得到所有样本、所有类别的平均结果。
    """

    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    """把需要最大化的 Dice 系数转换为可最小化的损失。

    Dice 越高越好，而优化器默认最小化目标，因此使用 ``1 - Dice``。
    """

    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
