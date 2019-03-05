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


class UniqueClipBoxes(keras.layers.Layer):
    """
    统一的边框裁剪
    """

    def __init__(self, clip_box_shape, **kwargs):
        """

        :param clip_box_shape: 裁剪的边框形状，tuple(H,W,C)or tuple(H,W)
        :param kwargs:
        """
        self.clip_box_shape = clip_box_shape
        super(UniqueClipBoxes, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs: boxes [batch_size,N,(y1,x1,y2,x2)]
        :param kwargs:
        :return:
        """
        boxes = inputs
        wy1, wx1, wy2, wx2 = 0., 0., float(self.clip_box_shape[0]), float(self.clip_box_shape[1])
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)  # split后维数不变
        # 裁剪到窗口内
        y1 = tf.minimum(tf.maximum(y1, wy1), wy2)  # wy1<=y1<=wy2
        y2 = tf.minimum(tf.maximum(y2, wy1), wy2)
        x1 = tf.minimum(tf.maximum(x1, wx1), wx2)  # wx1<=x1<=wx2
        x2 = tf.minimum(tf.maximum(x2, wx1), wx2)

        boxes = tf.concat([y1, x1, y2, x2], axis=-1, name='unique_clip_boxes')
        return boxes

    def compute_output_shape(self, input_shape):
        return input_shape
