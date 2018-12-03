# -*- coding: utf-8 -*-
"""
Created on 2018/12/2 上午8:48

@author: mick.yi

anchor层，生成anchors

"""

import tensorflow as tf
import keras
import numpy as np


def generate_anchors(base_size, ratios, scales):
    """

    :param base_size: anchor的base_size,如：（64，64）
    :param ratios: 长宽比 shape:(M,)
    :param scales: 缩放比 shape:(N,)
    :return: （N*M,2)
    """
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
    H, W = shape
    ctr_x = (tf.range(W, dtype=tf.float32) + tf.constant(0.5, dtype=tf.float32)) * strides
    ctr_y = (tf.range(H, dtype=tf.float32) + tf.constant(0.5, dtype=tf.float32)) * strides

    ctr_x, ctr_y = tf.meshgrid(ctr_x, ctr_y)

    # 打平为1维,得到所有锚点的坐标
    ctr_x = tf.reshape(ctr_x, [-1])
    ctr_y = tf.reshape(ctr_y, [-1])
    #  (H*W,1,4)
    shifts = tf.expand_dims(tf.transpose(tf.stack([ctr_y, ctr_x, ctr_y, ctr_x])), axis=1)
    # (1,anchor_num,4)
    base_anchors = tf.expand_dims(tf.constant(base_anchors, dtype=tf.float32), axis=0)

    # (H*W,anchor_num,4)
    anchors = shifts + base_anchors
    # 转为(H*W*anchor*num,4) 返回
    return tf.reshape(anchors, [-1, 4])


class Anchor(keras.layers.Layer):
    def __init__(self, base_size, strides, ratios, scales):
        """

        :param base_size: anchor的base_size,如：64
        :param strides: 步长
        :param ratios: 长宽比
        :param scales: 缩放比
        """
        self.base_size = base_size
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        # base anchors数量
        self.num_anchors = len(ratios) * len(scales)

    def call(self, inputs, **kwargs):
        """

        :param inputs: 卷积层特征(锚点所在层)，shape：[batch_num,H,W,C]
        :param kwargs:
        :return:
        """
        features = inputs
        features_shape = tf.shape(features)

        base_anchors = generate_anchors(self.base_size, self.ratios, self.scales)
        anchors = shift(features_shape[1:3], base_anchors)
        # 扩展第一维，batch_num;每个样本都有相同的anchors
        return tf.tile(tf.expand_dims(anchors, axis=0), [features_shape[0], 1, 1])

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: [batch_num,H,W,C]
        :return:
        """
        # 计算所有的anchors数量
        total = np.prod(input_shape[1:3]) * self.num_anchors
        return [input_shape[0], total, 4]


if __name__ == '__main__':
    sess = tf.Session()
    achrs = generate_anchors(64, [1], [1, 2, 4])
    all_achrs = shift([3, 3], 32, achrs)
    print(sess.run(tf.shape(all_achrs)))
    print(sess.run(all_achrs))
