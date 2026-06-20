"""语义分割数据集读取与预处理。

分割任务中的每个样本由两部分组成：

- 原始图像：作为网络输入；
- 掩码图像：每个像素的值或颜色表示该像素所属类别。

本模块负责匹配图像与掩码、扫描掩码类别值、统一缩放和归一化，并将数据转换为
PyTorch 张量。
"""

import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    """根据文件扩展名加载图像。

    除常见图片格式外，也支持保存为 NumPy 数组的 ``.npy`` 文件和保存为
    PyTorch 张量的 ``.pt/.pth`` 文件。所有格式最终统一转换为 PIL Image。
    """

    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    """读取一个样本的掩码，并返回其中出现过的所有唯一像素值。

    参数:
        idx: 不包含扩展名的样本 ID。
        mask_dir: 掩码目录。
        mask_suffix: 掩码文件名后缀，例如 Carvana 数据集的 ``_mask``。

    灰度掩码返回一维类别值；RGB 掩码把每种颜色视为一个类别并返回颜色列表。
    """

    # glob 允许掩码采用 png、jpg、npy 等不同扩展名。
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        # 灰度掩码：每个不同灰度值代表一个类别。
        return np.unique(mask)
    elif mask.ndim == 3:
        # 彩色掩码：先展平空间维度，再统计不同的 RGB/多通道颜色。
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    """图像与掩码同名的通用语义分割数据集。

    例如 ``imgs/cat.png`` 对应 ``masks/cat.png``。如果掩码名称中额外包含
    后缀，可通过 ``mask_suffix`` 指定。
    """

    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        """建立文件索引并扫描整个数据集中的掩码类别。

        ``scale`` 只控制读取后的图像尺寸，不会修改磁盘上的原始文件。
        """

        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # 使用图像文件名（去掉扩展名）作为样本 ID，并忽略 .keep 等隐藏文件。
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        # 多进程并行扫描每张掩码中的唯一值。
        # 先确定完整的 mask_values，之后才能把任意原始像素值稳定映射成 0、1、2……类别索引。
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        # 合并所有样本的类别值并排序，确保映射顺序在整个数据集中保持一致。
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        """返回数据集包含的样本数量。"""

        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        """缩放并转换一张输入图像或掩码。

        图像处理结果:
            返回 ``[C, H, W]`` 浮点 NumPy 数组，像素通常归一化到 0～1。

        掩码处理结果:
            返回 ``[H, W]`` 整数数组，原始像素值被映射为连续类别索引。

        掩码必须使用最近邻插值，避免缩放时产生原数据中不存在的类别值；
        普通图像则使用双三次插值获得更平滑的视觉结果。
        """

        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # 输出掩码不保留原始灰度值/颜色，而是统一编码为从 0 开始的类别索引。
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    # 灰度掩码按单个像素值匹配。
                    mask[img == v] = i
                else:
                    # RGB 掩码要求一个像素的所有通道都与类别颜色一致。
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                # 灰度图缺少通道维，将 [H, W] 转为 [1, H, W]。
                img = img[np.newaxis, ...]
            else:
                # PIL/NumPy 使用 [H, W, C]，PyTorch 卷积使用 [C, H, W]。
                img = img.transpose((2, 0, 1))

            # 常见 8 位图像范围为 0～255；归一化后更适合神经网络训练。
            # 若数据本身已位于 0～1，则不重复缩放。
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        """读取第 ``idx`` 个图像/掩码对并返回张量字典。"""

        name = self.ids[idx]
        # 支持不同文件扩展名，但每个 ID 必须且只能匹配到一个图像和一个掩码。
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # 像素级监督要求图像和掩码严格对齐，因此二者原始尺寸必须一致。
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # 图像与掩码使用相同缩放比例，但插值方式和输出编码不同。
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        # copy() 避免 NumPy 负步长或只读内存引起 torch.as_tensor 警告。
        # contiguous() 保证数据在内存中连续，便于后续搬运和卷积计算。
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    """适配 Carvana 数据集命名规则的数据集。

    Carvana 中原图 ``xxx.jpg`` 对应的掩码名为 ``xxx_mask.gif``，因此只需在
    BasicDataset 基础上固定 ``mask_suffix='_mask'``。
    """

    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
