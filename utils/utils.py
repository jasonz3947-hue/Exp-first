"""项目中的简单可视化工具。"""

import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    """并排显示输入图像及各类别的二值掩码。

    参数:
        img: PIL 图像或 Matplotlib 可显示的数组。
        mask: 二维类别索引数组，值为 0、1、2……。

    第一个子图显示原图，后续子图分别显示 ``mask == i`` 的结果，便于观察
    每个类别在图像中的预测位置。
    """

    # 最大类别索引加 1 即类别数量。
    classes = mask.max() + 1
    # 创建“原图 + 每个类别掩码”所需数量的横向子图。
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        # 布尔数组中属于当前类别的像素为 True，其余为 False。
        ax[i + 1].imshow(mask == i)
    # 分割结果主要关注区域形状，因此隐藏坐标轴刻度。
    plt.xticks([]), plt.yticks([])
    plt.show()
