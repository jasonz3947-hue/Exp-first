"""语义分割验证指标。

训练或测试过程中调用 :func:`evaluate`，在不计算梯度的情况下统计 Dice、IoU、
Precision、Recall 和 95% Hausdorff Distance（HD95）。
"""

from typing import Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from tqdm import tqdm


def _hd95(
        mask_pred: np.ndarray,
        mask_true: np.ndarray,
        spacing: Optional[Sequence[float]] = None,
) -> float:
    """计算两个二值掩码边界之间的对称 HD95。

    ``spacing`` 未提供时，距离单位为像素。两张掩码均为空时距离为 0；只有一张
    为空时使用图像对角线作为有限惩罚值，避免整个数据集的平均值变成无穷大。
    """

    mask_pred = np.asarray(mask_pred, dtype=bool)
    mask_true = np.asarray(mask_true, dtype=bool)
    if mask_pred.shape != mask_true.shape:
        raise ValueError(f'Prediction and target shapes differ: {mask_pred.shape} vs {mask_true.shape}')

    pred_nonempty = bool(mask_pred.any())
    true_nonempty = bool(mask_true.any())
    if not pred_nonempty and not true_nonempty:
        return 0.0

    if spacing is None:
        spacing_array = np.ones(mask_pred.ndim, dtype=np.float64)
    else:
        spacing_array = np.asarray(spacing, dtype=np.float64)
        if spacing_array.shape != (mask_pred.ndim,):
            raise ValueError(
                f'spacing must contain {mask_pred.ndim} values, got {spacing_array.tolist()}'
            )

    if pred_nonempty != true_nonempty:
        extent = np.maximum(np.asarray(mask_pred.shape, dtype=np.float64) - 1, 0)
        return float(np.linalg.norm(extent * spacing_array))

    structure = generate_binary_structure(mask_pred.ndim, 1)
    pred_surface = np.logical_xor(
        mask_pred,
        binary_erosion(mask_pred, structure=structure, border_value=0),
    )
    true_surface = np.logical_xor(
        mask_true,
        binary_erosion(mask_true, structure=structure, border_value=0),
    )

    distance_to_true = distance_transform_edt(~true_surface, sampling=spacing_array)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_array)
    surface_distances = np.concatenate((
        distance_to_true[pred_surface],
        distance_to_pred[true_surface],
    ))
    return float(np.percentile(surface_distances, 95))


def _binary_metrics(
        mask_pred: np.ndarray,
        mask_true: np.ndarray,
        spacing: Optional[Sequence[float]] = None,
) -> dict:
    """计算一对二值掩码的四项指标。"""

    mask_pred = np.asarray(mask_pred, dtype=bool)
    mask_true = np.asarray(mask_true, dtype=bool)

    intersection = int(np.logical_and(mask_pred, mask_true).sum())
    pred_size = int(mask_pred.sum())
    true_size = int(mask_true.sum())

    if pred_size == 0 and true_size == 0:
        dice = iou = precision = recall = 1.0
    else:
        dice = 2.0 * intersection / (pred_size + true_size)
        union = pred_size + true_size - intersection
        iou = intersection / union
        # 无预测前景但存在真实前景时，将 Precision 记为 0。
        precision = intersection / pred_size if pred_size > 0 else 0.0
        # 无真实前景但产生了前景预测时，将 Recall 记为 0；两者皆空已在上面记为 1。
        recall = intersection / true_size if true_size > 0 else 0.0

    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'hd95': _hd95(mask_pred, mask_true, spacing),
    }


@torch.inference_mode()
def evaluate(
        net,
        dataloader,
        device,
        amp,
        spacing: Optional[Sequence[float]] = None,
        show_progress: bool = False,
):
    """计算验证/测试数据集的平均分割指标。

    每张图片的每个前景类别分别计算指标，最后进行等权宏平均。多分类任务忽略
    第 0 类背景；二分类单输出通道任务将 sigmoid 概率大于 0.5 的像素视为前景。

    参数:
        net: 待评估的分割网络。
        dataloader: 验证集或测试集 DataLoader，建议设置 ``drop_last=False``。
        device: 推理设备。
        amp: 是否启用自动混合精度。
        spacing: 各空间维度的像素间距，例如二维医学图像可传 ``(row_mm, col_mm)``。
        show_progress: 是否显示验证批次进度条，训练时默认关闭。

    返回:
        包含 ``dice``、``iou``、``precision``、``recall`` 和 ``hd95`` 的字典。
    """

    was_training = net.training
    net.eval()
    totals = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'hd95': 0.0}
    metric_count = 0
    num_batches = len(dataloader)

    try:
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(
                    dataloader,
                    total=num_batches,
                    desc='Validation round',
                    unit='batch',
                    leave=False,
                    disable=not show_progress,
            ):
                images = batch['image'].to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                masks_true = batch['mask'].to(device=device, dtype=torch.long)
                logits = net(images)

                if net.n_classes == 1:
                    assert masks_true.min() >= 0 and masks_true.max() <= 1, \
                        'True mask indices should be in [0, 1]'
                    masks_pred = (torch.sigmoid(logits).squeeze(1) > 0.5)
                    masks_pred_np = masks_pred.cpu().numpy()
                    masks_true_np = masks_true.bool().cpu().numpy()

                    mask_pairs = zip(masks_pred_np, masks_true_np)
                else:
                    assert masks_true.min() >= 0 and masks_true.max() < net.n_classes, \
                        f'True mask indices should be in [0, {net.n_classes})'
                    labels_pred_np = logits.argmax(dim=1).cpu().numpy()
                    labels_true_np = masks_true.cpu().numpy()
                    mask_pairs = (
                        (labels_pred_np[sample_index] == class_index,
                         labels_true_np[sample_index] == class_index)
                        for sample_index in range(labels_true_np.shape[0])
                        for class_index in range(1, net.n_classes)
                    )

                for mask_pred, mask_true in mask_pairs:
                    values = _binary_metrics(mask_pred, mask_true, spacing)
                    for name in totals:
                        totals[name] += values[name]
                    metric_count += 1
    finally:
        net.train(was_training)

    if metric_count == 0:
        return {name: 0.0 for name in totals}
    return {name: total / metric_count for name, total in totals.items()}
