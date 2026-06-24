"""Evaluate every test image separately and sort rows by HD95.

This script uses the same dataset loading, checkpoint metadata, mask-value
mapping, and metric implementation as ``test.py``/``evaluate.py``. It is useful
for finding the images that dominate the average HD95.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from evaluate import _binary_metrics
from test import infer_model_config, load_test_dataset, normalized_mask_value
from unet import UNet


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained U-Net one test image at a time'
    )
    parser.add_argument('--model', '-m', type=Path, required=True,
                        help='Path to the trained .pth checkpoint')
    parser.add_argument('--images', type=Path, default=Path('data/test_imgs'),
                        help='Directory containing test images')
    parser.add_argument('--masks', type=Path, default=Path('data/test_masks'),
                        help='Directory containing ground-truth test masks')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Image scaling factor; normally the same value used during training')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Foreground threshold for one-channel binary models')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision on CUDA')
    parser.add_argument('--spacing', type=float, nargs=2, metavar=('ROW', 'COL'),
                        help='Pixel spacing used by HD95, for example --spacing 0.5 0.5')
    parser.add_argument('--top-k', type=int, default=0,
                        help='Print only the worst K images by HD95; 0 prints all images')
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device):
    state_dict = torch.load(model_path, map_location=device)
    training_mask_values = state_dict.pop('mask_values', None)
    n_channels, n_classes, bilinear = infer_model_config(state_dict)

    model = UNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
    )
    model.load_state_dict(state_dict)
    model.to(device=device, memory_format=torch.channels_last)
    model.eval()

    return model, training_mask_values, n_channels, n_classes, bilinear


def apply_training_mask_values(dataset, training_mask_values):
    if training_mask_values is None:
        logging.warning(
            'The checkpoint has no mask_values metadata; '
            'using the class mapping inferred from the test masks.'
        )
        return

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


def mean_metric_dict(metric_dicts):
    keys = metric_dicts[0].keys()
    return {
        key: float(np.mean([values[key] for values in metric_dicts]))
        for key in keys
    }


def foreground_area(labels: np.ndarray) -> int:
    return int(np.asarray(labels != 0, dtype=bool).sum())


@torch.inference_mode()
def evaluate_image(
        model,
        image: torch.Tensor,
        mask_true: torch.Tensor,
        device: torch.device,
        threshold: float,
        amp: bool,
        spacing: Optional[Sequence[float]],
) -> dict:
    image_batch = image.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    mask_true = mask_true.to(device=device, dtype=torch.long)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        logits = model(image_batch)

    if model.n_classes == 1:
        assert mask_true.min() >= 0 and mask_true.max() <= 1, \
            'True mask indices should be in [0, 1]'
        labels_pred = (torch.sigmoid(logits).squeeze(0).squeeze(0) > threshold)
        labels_true = mask_true.bool()
        metrics = _binary_metrics(
            labels_pred.cpu().numpy(),
            labels_true.cpu().numpy(),
            spacing,
        )
        return {
            **metrics,
            'pred_pixels': int(labels_pred.sum().item()),
            'true_pixels': int(labels_true.sum().item()),
        }

    assert mask_true.min() >= 0 and mask_true.max() < model.n_classes, \
        f'True mask indices should be in [0, {model.n_classes})'
    labels_pred = logits.argmax(dim=1).squeeze(0)

    class_metrics = []
    labels_pred_np = labels_pred.cpu().numpy()
    labels_true_np = mask_true.cpu().numpy()
    for class_index in range(1, model.n_classes):
        class_metrics.append(_binary_metrics(
            labels_pred_np == class_index,
            labels_true_np == class_index,
            spacing,
        ))

    metrics = mean_metric_dict(class_metrics)
    return {
        **metrics,
        'pred_pixels': foreground_area(labels_pred_np),
        'true_pixels': foreground_area(labels_true_np),
    }


def print_config(args, model, n_channels, n_classes, bilinear, dataset, device):
    hd95_unit = 'spacing units' if args.spacing else 'pixels'
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('\nEvaluation parameters')
    print(f'Model:      {args.model}')
    print(f'Images:     {args.images}')
    print(f'Masks:      {args.masks}')
    print(f'Device:     {device}')
    print(f'Params:     {total_params:,} total / {trainable_params:,} trainable')
    print(f'Channels:   {n_channels}')
    print(f'Classes:    {n_classes}')
    print(f'Bilinear:   {bilinear}')
    print(f'Scale:      {args.scale:g}')
    print(f'Threshold:  {args.threshold:g}')
    print(f'AMP:        {args.amp and device.type == "cuda"}')
    print(f'HD95 unit:  {hd95_unit}')
    print(f'Images N:   {len(dataset)}')


def print_results(rows, top_k):
    rows = sorted(rows, key=lambda row: row['hd95'], reverse=True)
    if top_k > 0:
        rows = rows[:top_k]

    print('\nPer-image metrics sorted by HD95 descending')
    print(
        f'{"rank":>4}  {"id":<24}  {"dice":>8}  {"iou":>8}  '
        f'{"prec":>8}  {"recall":>8}  {"hd95":>10}  '
        f'{"pred_px":>9}  {"true_px":>9}'
    )
    print('-' * 104)
    for rank, row in enumerate(rows, start=1):
        print(
            f'{rank:>4}  {row["id"]:<24}  '
            f'{row["dice"]:>8.6f}  {row["iou"]:>8.6f}  '
            f'{row["precision"]:>8.6f}  {row["recall"]:>8.6f}  '
            f'{row["hd95"]:>10.6f}  '
            f'{row["pred_pixels"]:>9}  {row["true_pixels"]:>9}'
        )


def print_average(rows):
    averaged = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ('dice', 'iou', 'precision', 'recall', 'hd95')
    }
    print('\nAverage over all evaluated images')
    print(f"Dice:      {averaged['dice']:.6f}")
    print(f"IoU:       {averaged['iou']:.6f}")
    print(f"Precision: {averaged['precision']:.6f}")
    print(f"Recall:    {averaged['recall']:.6f}")
    print(f"HD95:      {averaged['hd95']:.6f}")


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not args.model.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.model}')
    if not args.images.is_dir():
        raise FileNotFoundError(f'Test image directory not found: {args.images}')
    if not args.masks.is_dir():
        raise FileNotFoundError(f'Test mask directory not found: {args.masks}')
    if args.top_k < 0:
        raise ValueError('--top-k cannot be negative')
    if not 0 < args.threshold < 1:
        raise ValueError('--threshold must be between 0 and 1')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = args.amp and device.type == 'cuda'

    model, training_mask_values, n_channels, n_classes, bilinear = load_model(
        args.model,
        device,
    )
    dataset = load_test_dataset(args.images, args.masks, args.scale)
    apply_training_mask_values(dataset, training_mask_values)

    print_config(args, model, n_channels, n_classes, bilinear, dataset, device)

    rows = []
    for index in tqdm(range(len(dataset)), desc='Evaluating images', unit='image'):
        sample = dataset[index]
        values = evaluate_image(
            model=model,
            image=sample['image'],
            mask_true=sample['mask'],
            device=device,
            threshold=args.threshold,
            amp=amp,
            spacing=args.spacing,
        )
        rows.append({
            'id': dataset.ids[index],
            **values,
        })

    print_results(rows, args.top_k)
    print_average(rows)


if __name__ == '__main__':
    main()
