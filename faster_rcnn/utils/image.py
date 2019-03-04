# -*- coding: utf-8 -*-
"""
Created on 2018/12/15 下午5:42

@author: mick.yi

图像处理工具类

"""
import skimage
from skimage import io, transform
import numpy as np


def load_image(image_path):
    """
    加载图像
    :param image_path: 图像路径
    :return: [h,w,3] numpy数组
    """
    # Load image
    image = io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # 删除alpha通道
    return image[..., :3]


def load_image_gt(image_id, image_path, output_size, gt_boxes=None):
    """
    加载图像，生成训练输入大小的图像，并调整GT 边框，返回相关元数据信息
    :param image_id: 图像编号id
    :param image_path: 图像路径
    :param output_size: 标量，图像输出尺寸，及网络输入到高度或宽度(默认长宽相等)
    :param gt_boxes: GT 边框 [N,(y1,x1,y2,x2)]
    :return:
    image: (H,W,3)
    image_meta: 元数据信息，详见compose_image_meta
    gt_boxes：图像缩放及padding后对于的GT 边框坐标 [N,(y1,x1,y2,x2)]
    """
    # 加载图像
    image = load_image(image_path)
    original_shape = image.shape
    # resize图像，并获取相关元数据信息
    image, window, scale, padding = resize_image(image, output_size)

    # 组合元数据信息
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale)
    # 根据缩放及padding调整GT边框
    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        gt_boxes = adjust_box(gt_boxes, padding, scale)

    return image, image_meta, gt_boxes


def resize_image(image, max_dim):
    """
    缩放图像为正方形，指定长边大小，短边padding;
    :param image: numpy 数组(H,W,3)
    :param max_dim: 长边大小
    :return: 缩放后的图像,元素图像的宽口位置，缩放尺寸，padding
    """
    image_dtype = image.dtype
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)  # 缩放尺寸
    image = transform.resize(image, (round(h * scale), round(w * scale)),
                             order=1, mode='constant', cval=0, clip=True, preserve_range=True)
    h, w = image.shape[:2]
    # 计算padding
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)  #
    return image.astype(image_dtype), window, scale, padding


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale]  # size=1
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32)
    }


def adjust_box(boxes, padding, scale):
    """
    根据填充和缩放因子，调整bocompose_image_metaxes的值
    :param boxes: numpy 数组; GT boxes [N,(y1,x1,y2,x2)]
    :param padding: [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    :param scale: 缩放因子
    :return:
    """
    boxes = boxes * scale
    boxes[:, 0::2] += padding[0][0]  # 高度padding
    boxes[:, 1::2] += padding[1][0]  # 宽度padding
    return boxes
