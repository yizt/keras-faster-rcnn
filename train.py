# -*- coding: utf-8 -*-
"""
Created on 2018/12/16 上午9:30

@author: mick.yi

训练frcnn

"""

import random
import numpy as np
from faster_rcnn.config import current_config as config
from faster_rcnn.preprocess.pascal_voc import get_voc_data
from faster_rcnn.utils.image import load_image_gt, fix_num_pad
from faster_rcnn.layers.models import rpn_net, compile


def generator(all_image_info, batch_size):
    image_length = len(all_image_info)
    id_list = range(image_length)
    while True:
        ids = random.sample(id_list, batch_size)
        batch_image = []
        batch_image_meta = []
        batch_class_ids = []
        batch_bbox = []

        for id in ids:
            image, image_meta, class_ids, bbox = load_image_gt(
                config, all_image_info[id], id)
            batch_image.append(image)
            batch_image_meta.append(image_meta)
            # gt个数固定
            batch_class_ids.append([[0, 1, 0]] * len(class_ids) + [[0, 0, -1]] * (50 - len(class_ids)))
            batch_bbox.append(fix_num_pad(bbox, 50))

        yield [np.asarray(batch_image),
               np.asarray(batch_class_ids),
               np.asarray(batch_bbox)], None


if __name__ == '__main__':
    voc_path = '/Users/yizuotian/dataset/VOCdevkit/'
    all_img_info, classes_count, class_mapping = get_voc_data(voc_path)
    m = rpn_net((224, 224, 3), 50)
    compile(m, config, 0.01, 0.9)
    m.fit_generator(generator(all_img_info, 32),
                    epochs=3,
                    steps_per_epoch=len(all_img_info) // 32)
