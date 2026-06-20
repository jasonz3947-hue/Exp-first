"""验证集评估工具。

训练过程中调用 :func:`evaluate`，在不计算梯度的情况下统计模型在验证集上的
平均 Dice 分数。Dice 越接近 1，表示预测区域与真实区域的重合程度越高。
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    """计算模型在一个验证数据加载器上的平均 Dice 分数。

    参数:
        net: 待评估的分割网络。
        dataloader: 验证集 DataLoader。
        device: 推理设备。
        amp: 是否启用自动混合精度。

    返回:
        所有验证批次 Dice 分数的平均值。

    ``torch.inference_mode`` 比 ``no_grad`` 更彻底地关闭自动求导相关状态，
    适合纯推理/验证过程。
    """

    # 切换到评估模式，使 BatchNorm、Dropout 等层采用推理行为。
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # 遍历验证集；AMP 设置与训练阶段保持一致。
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # 图像采用浮点数和 channels_last 布局；标签保持整数类别索引。
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 输出为每个类别对应的原始 logits。
            mask_pred = net(image)

            if net.n_classes == 1:
                # 单类别任务的真实标签只能包含 0 和 1。
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # sigmoid 将 logits 转为概率，0.5 阈值再将其离散化为预测掩码。
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # 二分类时直接计算预测前景与真实前景的 Dice。
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # 多分类标签必须是 [0, n_classes) 范围内的整数。
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # Dice 计算需要 one-hot 张量，形状从 [N, H, W] 转为 [N, C, H, W]。
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # 第 0 类通常表示背景，只对前景类别统计 Dice。
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    # 恢复训练模式，避免调用评估后影响后续训练批次。
    net.train()
    # 空验证集时分母至少为 1，防止除零。
    return dice_score / max(num_val_batches, 1)
