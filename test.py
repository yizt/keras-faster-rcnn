# -*- coding: utf-8 -*-
"""
Created on 2018/11/13 9:57

@author: mick.yi

"""
import numpy as np
from faster_rcnn.preprocess.pascal_voc import PascalVoc, get_voc_data
from faster_rcnn.preprocess.input import load_image_gt, parse_image_meta
from faster_rcnn.config import VOCConfig
import tensorflow as tf


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

    image, image_meta, class_ids, bbox = load_image_gt(voc, VOCConfig(), 3)
    print(parse_image_meta(np.asarray([image_meta])))


if __name__ == '__main__':
    sess = tf.Session()

    x = tf.ones([10])
    y = tf.nn.top_k(tf.Variable([1, 3, 7, 4]) * -1, 4, sorted=False)
    x = tf.sparse_to_dense(y[0] * -1, [10], 1)

    sess.run(tf.global_variables_initializer())

    print(sess.run(y[0]))
    print(sess.run(x))
    print(sess.run(tf.stack([tf.ones([0]), tf.ones([0])], axis=1)))

