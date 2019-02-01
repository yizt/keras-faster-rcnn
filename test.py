# -*- coding: utf-8 -*-
"""
Created on 2018/11/13 9:57

@author: mick.yi

"""
import numpy as np
from faster_rcnn.preprocess.pascal_voc import PascalVoc, get_voc_data
from faster_rcnn.utils.image import load_image_gt, parse_image_meta
from faster_rcnn.config import current_config as config
import tensorflow as tf
from faster_rcnn.layers.models import rpn_net
from faster_rcnn.utils import visualize, np_utils


def clean_name(name):
    """Returns a shorter version of object names for cleaner display."""
    return ",".join(name.split(",")[:1])


def voc_test():
    imgs, class_count, class_map = get_voc_data('/Users/yizuotian/dataset/VOCdevkit')
    print(class_count)
    print(class_map)
    print(len(class_map))

    voc = PascalVoc()
    voc.load_voc('/Users/yizuotian/dataset/VOCdevkit')
    voc.prepare()
    print(voc.class_ids)
    print(voc.class_info)
    print(voc.class_from_source_map)
    print(voc.image_info[1])

    image, image_meta, class_ids, bbox = load_image_gt(voc, config, 3)
    print(parse_image_meta(np.asarray([image_meta])))


if __name__ == '__main__':
    # from tensorflow.python import debug as tf_debug
    # import keras.backend as K
    #
    # sess = K.get_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # K.set_session(sess)
    # 加载数据
    all_img_info, classes_count, class_mapping = get_voc_data(config.voc_path)
    id = 3
    image, image_meta, class_ids, bbox = load_image_gt(config, all_img_info[id], id)
    print(image.shape)
    # 加载网络
    m = rpn_net((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3), 50, 1, stage='test')
    m.load_weights(config.weights, by_name=True)
    m.summary()
    boxes, scores = m.predict([np.expand_dims(image, axis=0), np.expand_dims(image_meta, axis=0)])

    boxes = np_utils.remove_pad(boxes)
    scores = np_utils.remove_pad(scores)
    visualize.display_instances(image, boxes,
                                [1] * len(boxes),
                                ['fg'] * len(boxes),
                                scores=np.squeeze(scores, axis=1))

    print(boxes)
    print(scores)
