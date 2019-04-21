# -*- coding: utf-8 -*-
"""
Created on 2018/12/2 上午8:48

@author: mick.yi

anchor层，生成anchors

"""

import tensorflow as tf
import keras
import numpy as np


def generate_anchors(heights, widths, base_size, ratios, scales):
    """
    :param heights: anchor高度列表
    :param widths: anchor宽度列表
    根据基准尺寸、长宽比、缩放比生成边框
    :param base_size: anchor的base_size,如：64
    :param ratios: 长宽比 shape:(M,)
    :param scales: 缩放比 shape:(N,)
    :return: （N*M,(y1,x1,y2,x2))
    """
    if heights is not None:
        h = np.array(heights, np.float32)
        w = np.array(widths, np.float32)
    else:
        ratios = np.expand_dims(np.array(ratios), axis=1)  # (N,1)
        scales = np.expand_dims(np.array(scales), axis=0)  # (1,M)
        # 计算高度和宽度，形状为(N,M)
        h = np.sqrt(ratios) * scales * base_size
        w = 1.0 / np.sqrt(ratios) * scales * base_size

    # reshape为（N*M,1)
    h = np.reshape(h, (-1, 1))
    w = np.reshape(w, (-1, 1))

    return np.hstack([-0.5 * h, -0.5 * w, 0.5 * h, 0.5 * w])


def shift(shape, strides, base_anchors):
    """
    根据feature map的长宽，生成所有的anchors
    :param shape: （H,W)
    :param strides: 步长
    :param base_anchors:所有的基准anchors，(anchor_num,4)
    :return:
    """
    H, W = shape[0], shape[1]
    print("shape:{}".format(shape))
    ctr_x = (tf.cast(tf.range(W), tf.float32) + tf.constant(0.5, dtype=tf.float32)) * strides
    ctr_y = (tf.cast(tf.range(H), tf.float32) + tf.constant(0.5, dtype=tf.float32)) * strides

    ctr_x, ctr_y = tf.meshgrid(ctr_x, ctr_y)

    # 打平为1维,得到所有锚点的坐标
    ctr_x = tf.reshape(ctr_x, [-1])
    ctr_y = tf.reshape(ctr_y, [-1])
    #  (H*W,1,4)
    shifts = tf.expand_dims(tf.stack([ctr_y, ctr_x, ctr_y, ctr_x], axis=1), axis=1)
    # (1,anchor_num,4)
    base_anchors = tf.expand_dims(tf.constant(base_anchors, dtype=tf.float32), axis=0)

    # (H*W,anchor_num,4)
    anchors = shifts + base_anchors
    # 转为(H*W*anchor_num,4)
    anchors = tf.reshape(anchors, [-1, 4])
    # 丢弃越界的anchors;   步长*feature map的高度就是图像高度
    is_valid_anchors = tf.logical_and(tf.less_equal(anchors[:, 2], tf.cast(strides * H, tf.float32)),
                                      tf.logical_and(tf.less_equal(anchors[:, 3], tf.cast(strides * W, tf.float32)),
                                                     tf.logical_and(tf.greater_equal(anchors[:, 0], 0),
                                                                    tf.greater_equal(anchors[:, 1], 0))))
    return tf.reshape(anchors, [-1, 4]), is_valid_anchors


class Anchor(keras.layers.Layer):
    def __init__(self, heights=None, widths=None, base_size=None, ratios=None, scales=None, strides=None, **kwargs):
        """
        :param heights: anchor高度列表
        :param widths: anchor宽度列表
        :param base_size: anchor的base_size,如：64
        :param ratios: 长宽比; 如 [1,1/2,2]
        :param scales: 缩放比: 如 [1,2,4]
        :param strides: 步长,一般为base_size的四分之一
        """
        self.heights = heights
        self.widths = widths
        self.base_size = base_size
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        # base anchors数量
        self.num_anchors = len(heights) if heights is not None else len(ratios) * len(scales)
        super(Anchor, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs：输入
        input[0]: 卷积层特征(锚点所在层)，shape：[batch_size,H,W,C]
        input[1]: 图像的元数据信息, shape: [batch_size, 12 ];
        :param kwargs:
        :return:
        """
        features = inputs
        features_shape = tf.shape(features)
        print("feature_shape:{}".format(features_shape))

        base_anchors = generate_anchors(self.heights, self.widths, self.base_size, self.ratios, self.scales)
        anchors, anchors_tag = shift(features_shape[1:3], self.strides, base_anchors)
        # 扩展第一维，batch_size;每个样本都有相同的anchors
        anchors = tf.tile(tf.expand_dims(anchors, axis=0), [features_shape[0], 1, 1])
        anchors_tag = tf.tile(tf.expand_dims(anchors_tag, axis=0), [features_shape[0], 1])
        return anchors, anchors_tag

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: [batch_size,H,W,C]
        :return:
        """
        # 计算所有的anchors数量
        total = np.prod(input_shape[1:3]) * self.num_anchors
        # total = 49 * self.num_anchors
        return [(input_shape[0], total, 4),
                (input_shape[0], total)]


if __name__ == '__main__':
    sess = tf.Session()
    achrs = generate_anchors(64, [1], [1, 2, 4])
    print(achrs)
    all_achrs = shift([3, 3], 32, achrs)
    print(sess.run(tf.shape(all_achrs)))
    print(sess.run(all_achrs))
