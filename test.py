"""在独立测试集上评估训练好的 U-Net。

测试图片和真实掩码必须一一对应。默认目录结构：

    data/test_imgs/
    data/test_masks/

普通数据集要求图片与掩码主文件名相同，例如 ``001.jpg`` 对应
``001.png``；Carvana 格式也受支持，例如 ``001.jpg`` 对应
``001_mask.gif``。

运行示例：

    python test.py --model checkpoints/checkpoint_epoch104.pth
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained U-Net on a test set')
    parser.add_argument('--model', '-m', type=Path, required=True,
                        help='Path to the trained .pth checkpoint')
    parser.add_argument('--images', type=Path, default=Path('data/test_imgs'),
                        help='Directory containing test images')
    parser.add_argument('--masks', type=Path, default=Path('data/test_masks'),
                        help='Directory containing ground-truth test masks')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Number of test images per batch')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Image scaling factor; normally the same value used during training')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision on CUDA')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--spacing', type=float, nargs=2, metavar=('ROW', 'COL'),
                        help='Pixel spacing used by HD95, for example --spacing 0.5 0.5')
    return parser.parse_args()


def load_test_dataset(images_dir: Path, masks_dir: Path, scale: float):
    """按 Carvana 命名规则加载；不匹配时退回到同名图片/掩码规则。"""

    try:
        return CarvanaDataset(images_dir, masks_dir, scale)
    except (AssertionError, RuntimeError, IndexError):
        return BasicDataset(images_dir, masks_dir, scale)


def infer_model_config(state_dict):
    """从当前项目保存的 state_dict 中恢复网络结构参数。"""

    input_weight = state_dict.get('inc.double_conv.0.weight')
    output_weight = state_dict.get('outc.conv.weight')
    if input_weight is None or output_weight is None:
        raise ValueError(
            'The checkpoint does not match this U-Net implementation: '
            'required input/output convolution weights were not found.'
        )

    n_channels = input_weight.shape[1]
    n_classes = output_weight.shape[0]
    bilinear = 'up1.up.weight' not in state_dict
    return n_channels, n_classes, bilinear


def normalized_mask_value(value):
    """把灰度值或 RGB 列表转换为可比较的形式。"""

    return tuple(value) if isinstance(value, list) else value


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not args.model.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.model}')
    if not args.images.is_dir():
        raise FileNotFoundError(f'Test image directory not found: {args.images}')
    if not args.masks.is_dir():
        raise FileNotFoundError(f'Test mask directory not found: {args.masks}')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be at least 1')
    if args.workers < 0:
        raise ValueError('--workers cannot be negative')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device %s', device)
    logging.info('Loading checkpoint %s', args.model)

    state_dict = torch.load(args.model, map_location=device)
    training_mask_values = state_dict.pop('mask_values', None)
    n_channels, n_classes, bilinear = infer_model_config(state_dict)

    model = UNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
    )
    model.load_state_dict(state_dict)
    model.to(device=device, memory_format=torch.channels_last)

    dataset = load_test_dataset(args.images, args.masks, args.scale)

    # 测试掩码必须使用训练阶段相同的“原始像素值 -> 类别索引”映射。
    if training_mask_values is not None:
        training_values = {
            normalized_mask_value(value) for value in training_mask_values
        }
        unknown_values = [
            value for value in dataset.mask_values
            if normalized_mask_value(value) not in training_values
        ]
        if unknown_values:
            raise ValueError(
                f'Test masks contain values not seen during training: {unknown_values}. '
                f'Training mask values: {training_mask_values}'
            )
        dataset.mask_values = training_mask_values
    else:
        logging.warning(
            'The checkpoint has no mask_values metadata; '
            'using the class mapping inferred from the test masks.'
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
    )

    logging.info(
        'Testing %d images | channels=%d | classes=%d | bilinear=%s | scale=%g',
        len(dataset),
        n_channels,
        n_classes,
        bilinear,
        args.scale,
    )
    metrics = evaluate(
        model,
        loader,
        device,
        amp=args.amp and device.type == 'cuda',
        spacing=args.spacing,
        show_progress=True,
    )

    print('\nTest set metrics')
    print(f"Dice:      {metrics['dice']:.6f}")
    print(f"IoU:       {metrics['iou']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")
    hd95_unit = 'spacing units' if args.spacing else 'pixels'
    print(f"HD95:      {metrics['hd95']:.6f} {hd95_unit}")


if __name__ == '__main__':
    main()
