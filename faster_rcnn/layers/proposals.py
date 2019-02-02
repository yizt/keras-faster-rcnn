# -*- coding: utf-8 -*-
"""
   File Name：     proposals
   Description :  proposals候选框生成
   Author :       mick.yi
   date：          2019/2/1
"""
import keras
import tensorflow as tf
from faster_rcnn.utils import tf_utils


def apply_regress(deltas, anchors):
    """
    应用回归目标到边框
    :param deltas: 回归目标[N,(dy, dx, dh, dw)]
    :param anchors: anchor boxes[N,(y1,x1,y2,x2)]
    :return:
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    # 中心点坐标
    cy = (anchors[:, 2] + anchors[:, 0]) * 0.5
    cx = (anchors[:, 3] + anchors[:, 1]) * 0.5

    # 回归系数
    deltas *= tf.constant([0.1, 0.1, 0.2, 0.2])
    dy, dx, dh, dw = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 中心坐标回归
    cy += dy * h
    cx += dx * w
    # 高度和宽度回归
    h *= tf.exp(dh)
    w *= tf.exp(dw)

    # 转为y1,x1,y2,x2
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    y2 = cy + h * 0.5
    x2 = cx + w * 0.5

    return tf.stack([y1, x1, y2, x2], axis=1)


def nms(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=0.05, name=None):
    """
    非极大抑制
    :param boxes:
    :param scores:
    :param max_output_size:
    :param iou_threshold:
    :param score_threshold:
    :param name:
    :return: 检测边框、边框得分
    """
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold, score_threshold, name)  # 一维索引
    output_boxes = tf.gather(boxes, indices)  # (M,4)
    class_scores = tf.expand_dims(tf.gather(scores, indices), axis=1)  # 扩展到二维(M,1)
    # padding到固定大小
    return tf_utils.pad_to_fixed_size(output_boxes, max_output_size), \
           tf_utils.pad_to_fixed_size(class_scores, max_output_size)


class RpnToProposal(keras.layers.Layer):
    """
    生成候选框
    """

    def __init__(self, batch_size, score_threshold=0.05, output_box_num=300, iou_threshold=0.5, **kwargs):
        """

        :param batch_size: batch_size
        :param score_threshold: 分数阈值
        :param output_box_num: 生成proposal 边框数量
        :param iou_threshold: nms iou阈值
        """
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.output_box_num = output_box_num
        self.iou_threshold = iou_threshold
        super(RpnToProposal, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        应用边框回归，并使用nms生成最后的边框
        :param inputs:
        inputs[0]: deltas, [N,(dy,dx,dh,dw)]
        inputs[1]: class logits [N,num_classes]
        inputs[2]: anchors [N,(y1,x1,y2,x2)]
        :param kwargs:
        :return:
        """
        deltas = inputs[0]
        class_logits = inputs[1]
        anchors = inputs[2]
        # 转为分类评分
        class_scores = tf.nn.softmax(logits=class_logits, axis=-1)[..., -1]  # 最后一类为前景 (N,)

        # 应用边框回归
        proposals = tf_utils.batch_slice([deltas, anchors], lambda x, y: apply_regress(x, y), self.batch_size)

        # # 非极大抑制
        outputs = tf_utils.batch_slice([proposals, class_scores],
                                       lambda x, y: nms(x, y,
                                                        max_output_size=self.output_box_num,
                                                        iou_threshold=self.iou_threshold,
                                                        score_threshold=self.score_threshold),
                                       self.batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        """
        注意多输出，call返回值必须是列表
        :param input_shape:
        :return:
        """
        return [(input_shape[0][0], self.output_box_num, 4 + 1),
                (input_shape[0][0], self.output_box_num, 1 + 1)]
