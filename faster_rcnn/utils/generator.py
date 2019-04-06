# -*- coding: utf-8 -*-
"""
   File Name：     generator
   Description :  生成器
   Author :       mick.yi
   date：          2019/3/4
"""
import numpy as np
from faster_rcnn.utils import np_utils, image as image_utils


class Generator(object):
    def __init__(self, annotation_list, input_shape, batch_size=1, max_gt_num=50,
                 horizontal_flip=False, random_crop=False,
                 **kwargs):
        """

        :param annotation_list:
        :param input_shape:
        :param batch_size:
        :param max_gt_num:
        :param horizontal_flip:
        :param random_crop:
        :param kwargs:
        """
        self.input_shape = input_shape
        self.annotation_list = annotation_list
        self.batch_size = batch_size
        self.max_gt_num = max_gt_num
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        self.size = len(annotation_list)
        super(Generator, self).__init__(**kwargs)

    def gen(self):
        while True:
            images = np.zeros((self.batch_size,) + self.input_shape, dtype=np.float32)
            image_metas = np.zeros((self.batch_size, 12), dtype=np.float32)
            batch_gt_boxes = np.zeros((self.batch_size, self.max_gt_num, 5), dtype=np.float32)
            batch_gt_class_ids = np.ones((self.batch_size, self.max_gt_num, 2), dtype=np.uint8)
            # 随机选择
            indices = np.random.choice(self.size, self.batch_size, replace=False)
            for i, index in enumerate(indices):
                # 加载图像
                image = image_utils.load_image(self.annotation_list[index]['filepath'])

                # resize图像
                images[i], image_metas[i], gt_boxes = image_utils.resize_image_and_gt(image,
                                                                                      self.input_shape[0],
                                                                                      self.annotation_list[index][
                                                                                          'boxes'])
                # pad gt到固定个数
                batch_gt_boxes[i] = np_utils.pad_to_fixed_size(gt_boxes, self.max_gt_num)
                batch_gt_class_ids[i] = np_utils.pad_to_fixed_size(
                    np.expand_dims(self.annotation_list[index]['labels'], axis=1),
                    self.max_gt_num)

            yield {"input_image": images,
                   "input_image_meta": image_metas,
                   "input_gt_boxes": batch_gt_boxes,
                   "input_gt_class_ids": batch_gt_class_ids}, None

    def gen_val(self):
        """
        评估生成器
        :return:
        """
        for idx, image_info in enumerate(self.annotation_list):
            # 加载图像
            image = image_utils.load_image(self.annotation_list[idx]['filepath'])
            image, image_meta, _ = image_utils.resize_image_and_gt(image,
                                                                   self.input_shape[0])

            if idx % 200 == 0:
                print("开始预测:{}张图像".format(idx))
            yield {"input_image": np.asarray([image]),
                   "input_image_meta": np.asarray([image_meta])}
