# -*- coding: utf-8 -*-
"""
   File Name：     detect_boxes
   Description :  根据proposals以及rcnn的回归参数生成最终的检测边框
   Author :       mick.yi
   date：          2019/2/14
"""
import keras
import tensorflow as tf
from faster_rcnn.utils import tf_utils
from faster_rcnn.utils.tf_utils import apply_regress, nms


class ProposalToDetectBox(keras.layers.Layer):
    """
    根据候选框生成最终的检测框（与RpnToProposal基本一致,为避免名称歧义，重新复制了一份)
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
        super(ProposalToDetectBox, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        应用边框回归，并使用nms生成最后的边框
        :param inputs:
        inputs[0]: deltas, [batch_size,N,num_classes,(dy,dx,dh,dw)]   N是proposal数量; 注意这里的deltas是类别相关的
        inputs[1]: class logits [batch_size,N,num_classes]
        inputs[2]: proposals [batch_size,N,(y1,x1,y2,x2,tag)]
        :param kwargs:
        :return:
        """
        deltas = inputs[0]
        class_logits = inputs[1]
        proposals = inputs[2][..., :-1]  # 去除tag列
        # 转为分类评分
        class_scores = tf.nn.softmax(logits=class_logits, axis=-1)  # [batch_size,N,num_classes]
        fg_scores = tf.reduce_max(class_scores[..., 1:], axis=-1)  # 第一类为背景 (batch_size,N)

        # 应用边框回归
        detect_boxes = tf_utils.batch_slice([deltas, proposals], lambda x, y: apply_regress(x, y), self.batch_size)

        # # 非极大抑制
        outputs = tf_utils.batch_slice([detect_boxes, fg_scores, class_logits],
                                       lambda x, y, z: nms(x, y, z,
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
                (input_shape[0][0], self.output_box_num, 1 + 1),
                (input_shape[0][0], self.output_box_num, input_shape[1][-1])]
