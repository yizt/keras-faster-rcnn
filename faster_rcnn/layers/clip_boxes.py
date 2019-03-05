# -*- coding: utf-8 -*-
"""
   File Name：     clip_boxes
   Description :  边框裁剪层
   Author :       mick.yi
   date：          2019/3/5
"""

import keras
import tensorflow as tf
from ..utils import tf_utils


class ClipBoxes(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ClipBoxes, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        inputs[0]: boxes  [batch_size,N,(y1,x1,y2,x2)]
        inputs[1]: windows [batch_size,(y1,x1,y2,x2)]   一个batch一个window
        :param kwargs:
        :return: 裁剪后的边框 clipped_boxes  [batch_size,N,(y1,x1,y2,x2)]
        """
        boxes = inputs[0]
        windows = inputs[1]
        clipped_boxes = tf.map_fn(fn=lambda x: tf_utils.clip_boxes(*x),
                                  elems=[boxes, windows],
                                  dtype=tf.float32)
        return clipped_boxes

    def compute_output_shape(self, input_shape):
        return input_shape[0]
