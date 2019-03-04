# -*- coding: utf-8 -*-
"""
   File Name：     generator
   Description :  生成器
   Author :       mick.yi
   date：          2019/3/4
"""
import random
import numpy as np
from faster_rcnn.utils import np_utils, image as image_util


def generator(image_info_list, batch_size, max_output_dim, max_gt_num, stage='train'):
    """

    :param image_info_list: 字典列表
    :param batch_size:
    :param max_output_dim:
    :param max_gt_num:
    :param stage:
    :return:
    """
    image_length = len(image_info_list)
    id_list = range(image_length)
    while True:
        ids = random.sample(id_list, batch_size)
        batch_image = []
        batch_image_meta = []
        batch_class_ids = []
        batch_bbox = []

        for id in ids:
            image, image_meta, bbox = image_util.load_image_gt(id,
                                                               image_info_list[id]['filepath'],
                                                               max_output_dim,
                                                               image_info_list[id]['boxes'])
            batch_image.append(image)
            batch_image_meta.append(image_meta)
            if stage == 'train':
                # gt个数固定
                batch_class_ids.append(
                    np_utils.pad_to_fixed_size(np.expand_dims(image_info_list[id]['labels'], axis=1), max_gt_num))
                batch_bbox.append(np_utils.pad_to_fixed_size(bbox, max_gt_num))
        if stage == 'train':
            yield [np.asarray(batch_image),
                   np.asarray(batch_image_meta),
                   np.asarray(batch_class_ids),
                   np.asarray(batch_bbox)], None
        else:
            yield [np.asarray(batch_image),
                   np.asarray(batch_image_meta)]
