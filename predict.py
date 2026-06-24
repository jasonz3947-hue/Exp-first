"""使用训练好的 U-Net 权重对图像执行语义分割预测。

程序会读取一个或多个输入图像，将模型输出转换为掩码，并根据命令行参数
保存或显示结果。

运行示例：
    python predict.py --model checkpoints/checkpoint_epoch5.pth --input test.jpg --viz
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    """预测一张 PIL 图像对应的分割掩码。

    参数:
        net: 已加载权重的 U-Net。
        full_img: 原始 PIL 图像。
        device: 执行推理的设备。
        scale_factor: 输入网络前的缩放比例。
        out_threshold: 单类别模型将概率判定为前景的阈值。

    返回:
        二维 NumPy 数组，其中每个元素是对应像素的类别索引。
    """

    # 评估模式会关闭 Dropout，并让 BatchNorm 使用训练阶段统计量。
    net.eval()
    # 复用训练数据集的预处理逻辑，保证缩放、通道顺序和归一化方式一致。
    # 返回数组形状为 [C, H, W]。
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    # 模型要求批次维度，因此将单张图像扩展为 [1, C, H, W]。
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    # 推理阶段不需要保存梯度，可显著减少内存占用。
    with torch.no_grad():
        output = net(img).cpu()
        # 网络是在缩放后的图像上预测，需要把 logits 插值回原始图像尺寸。
        # PIL 的 size 为 (宽, 高)，F.interpolate 的目标尺寸为 (高, 宽)。
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            # 多分类：选择每个像素 logits 最大的通道作为预测类别。
            mask = output.argmax(dim=1)
        else:
            # 单类别：先经 sigmoid 转为前景概率，再使用阈值得到布尔掩码。
            mask = torch.sigmoid(output) > out_threshold

    # 移除批次维和长度为 1 的通道维，并转为 NumPy 供后续保存。
    return mask[0].long().squeeze().numpy()


def get_args():
    """定义并解析预测脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    """根据输入文件名生成默认输出文件名。

    例如 ``image.jpg`` 会生成 ``image_OUT.png``。若用户通过 ``--output``
    显式指定文件名，则直接使用用户提供的列表。
    """

    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    """把类别索引掩码还原为可保存的 PIL 图像。

    ``mask_values`` 来自训练集原始掩码，它可能是：

    - 灰度值列表，例如 ``[0, 255]``；
    - 二值列表 ``[0, 1]``；
    - RGB 颜色列表，例如 ``[[0, 0, 0], [255, 0, 0]]``。
    """

    # 根据原始掩码值的类型创建正确通道数和数据类型的输出数组。
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    # 如果传入的是每类一张概率/得分图，则先沿类别维选择最大值。
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # 将连续类别索引 0、1、2……映射回训练数据中的实际像素值或 RGB 颜色。
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    # 解析参数并配置统一的控制台日志格式。
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    # 输出文件数量应与输入图像数量一一对应。
    out_files = get_output_filenames(args)

    # 网络结构必须与训练权重的通道数、类别数和上采样方式一致。
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    # 优先使用 GPU 推理，无 CUDA 时自动回退到 CPU。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    # map_location 允许在与训练时不同的设备上加载权重。
    state_dict = torch.load(args.model, map_location=device)
    # 新版训练权重会包含原始掩码值；旧权重没有该字段时按二值掩码处理。
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    # 按输入顺序逐张预测，因此输出文件与输入文件通过索引对应。
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            # 将类别索引转换为原始掩码像素值后保存。
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            # Matplotlib 窗口关闭后才会继续处理下一张图。
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
