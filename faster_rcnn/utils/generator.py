# -*- coding: utf-8 -*-
"""
   File Name：     generator
   Description :  生成器
   Author :       mick.yi
   date：          2019/3/4
"""
import numpy as np
import random
from faster_rcnn.utils import np_utils, image as image_utils
from tensorflow.python import keras


def image_flip(image, gt_boxes):
    """
    水平翻转图像和gt boxes
    :param image: [H,W,3]
    :param gt_boxes: [n,(y1,x1,y2,x2)]
    :return:
    """
    # gt翻转
    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        x_min = image.shape[1] - gt_boxes[:, 3]  # x坐标关于图像中心对称x+x'=w
        x_max = image.shape[1] - gt_boxes[:, 1]
        # 左右位置互换,新的顺序为[y1,x_min,y2,x_max]
        gt_boxes = np.stack([gt_boxes[:, 0], x_min, gt_boxes[:, 2], x_max], axis=1)

    return image[:, ::-1, :], gt_boxes


def image_crop(image, gt_boxes):
    """
    随机裁剪图像和gt boxes
    :param image: [H,W,3]
    :param gt_boxes: [n,(y1,x1,y2,x2)]
    :return:
    """
    # gt坐标偏移
    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        # gt_boxes的窗口区域
        min_x, max_x = np.min(gt_boxes[:, 1::2]), np.max(gt_boxes[:, 1::2])
        min_y, max_y = np.min(gt_boxes[:, ::2]), np.max(gt_boxes[:, ::2])
        image, crop_window = image_utils.random_crop_image(image, [min_y, min_x, max_y, max_x])
        # print(image.shape,[min_y, min_x, max_y, max_x],crop_window)
        gt_boxes[:, ::2] -= crop_window[0]  # 高度偏移
        gt_boxes[:, 1::2] -= crop_window[1]  # 宽度偏移
    return image, gt_boxes


class Generator(keras.utils.data_utils.Sequence):
    def __init__(self, annotation_list, input_shape, mean_pixel, batch_size=1, max_gt_num=50,
                 horizontal_flip=False, random_crop=False,
                 **kwargs):
        """

        :param annotation_list:
        :param input_shape:
        :param mean_pixel:
        :param batch_size:
        :param max_gt_num:
        :param horizontal_flip:
        :param random_crop:
        :param kwargs:
        """
        self.input_shape = input_shape
        self.annotation_list = annotation_list
        self.mean_pixel = mean_pixel
        self.batch_size = batch_size
        self.max_gt_num = max_gt_num
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        self.size = len(annotation_list)
        super(Generator, self).__init__(**kwargs)

    def on_epoch_end(self):
        # 一个epoch重新打乱
        np.random.shuffle(self.annotation_list)

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, index):
        indices = np.arange(index * self.batch_size, (index + 1) * self.batch_size)
        images = np.zeros((self.batch_size,) + self.input_shape, dtype=np.float32)
        image_metas = np.zeros((self.batch_size, 12), dtype=np.float32)
        batch_gt_boxes = np.zeros((self.batch_size, self.max_gt_num, 5), dtype=np.float32)
        batch_gt_class_ids = np.ones((self.batch_size, self.max_gt_num, 2), dtype=np.uint8)
        for i, index in enumerate(indices):
            # 加载图像
            image = image_utils.load_image(self.annotation_list[index]['filepath'])
            # 数据增广:水平翻转、随机裁剪
            gt_boxes = self.annotation_list[index]['boxes'].copy()  # 不改变原来的
            if self.horizontal_flip and random.random() > 0.5:
                image, gt_boxes = image_flip(image, gt_boxes)
            if self.random_crop and random.random() > 0.5:
                image, gt_boxes = image_crop(image, gt_boxes)

            # resize图像
            images[i], image_metas[i], gt_boxes = image_utils.resize_image_and_gt(image,
                                                                                  self.input_shape[0],
                                                                                  gt_boxes)
            # pad gt到固定个数
            batch_gt_boxes[i] = np_utils.pad_to_fixed_size(gt_boxes, self.max_gt_num)
            batch_gt_class_ids[i] = np_utils.pad_to_fixed_size(
                np.expand_dims(self.annotation_list[index]['labels'], axis=1),
                self.max_gt_num)
        images = np.asarray(images, np.float32) - self.mean_pixel  # 减去均值

        return {"input_image": images,
                "input_image_meta": image_metas,
                "input_gt_boxes": batch_gt_boxes,
                "input_gt_class_ids": batch_gt_class_ids}, None


class TestGenerator(Generator):
    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        # 加载图像
        image = image_utils.load_image(self.annotation_list[index]['filepath'])
        image, image_meta, _ = image_utils.resize_image_and_gt(image,
                                                               self.input_shape[0])
        image = np.asarray(image, np.float32) - self.mean_pixel  # 减去均值
        if index % 200 == 0:
            print("开始预测:{}张图像".format(index))
        return {"input_image": np.asarray([image]),
                "input_image_meta": np.asarray([image_meta])}
