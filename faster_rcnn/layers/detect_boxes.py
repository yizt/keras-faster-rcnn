# -*- coding: utf-8 -*-
"""
   File Name：     detect_boxes
   Description :  根据proposals以及rcnn的回归参数生成最终的检测边框
   Author :       mick.yi
   date：          2019/2/14
"""
import tensorflow.python.keras as keras
import tensorflow as tf
from faster_rcnn.utils.tf_utils import apply_regress, pad_to_fixed_size


def detect_boxes(boxes, class_logits, max_output_size, iou_threshold=0.5, score_threshold=0.05, name=None):
    """
    使用类别相关的非极大抑制nms生成最终检测边框
    :param boxes: 形状为[num_boxes, 4]的二维浮点型Tensor.
    :param class_logits: 形状为[num_boxes,num_classes] 原始的预测类别
    :param max_output_size: 一个标量整数Tensor,表示通过非最大抑制选择的框的最大数量.
    :param iou_threshold: 浮点数,IOU 阈值
    :param score_threshold:  浮点数, 过滤低于阈值的边框
    :param name:
    :return: 检测边框、边框得分、边框类别、预测的logits
    """
    # 类别得分和预测类别
    class_scores = tf.reduce_max(tf.nn.softmax(class_logits, axis=-1)[:, 1:], axis=-1)  # [num_boxes]
    class_ids = tf.argmax(class_logits[:, 1:], axis=-1) + 1  # [num_boxes]
    # 过滤背景类别0
    keep = tf.where(class_ids > 0)  # 保留的索引号
    keep_class_scores = tf.gather_nd(class_scores, keep)
    keep_class_ids = tf.gather_nd(class_ids, keep)
    keep_boxes = tf.gather_nd(boxes, keep)
    keep_class_logits = tf.gather_nd(class_logits, keep)

    # 按类别nms
    unique_class_ids = tf.unique(class_ids)[0]

    def per_class_nms(class_id):
        # 当前类别的索引
        idx = tf.where(tf.equal(keep_class_ids, class_id))  # [n,1]
        cur_class_scores = tf.gather_nd(keep_class_scores, idx)
        cur_class_boxes = tf.gather_nd(keep_boxes, idx)

        indices = tf.image.non_max_suppression(cur_class_boxes,
                                               cur_class_scores,
                                               max_output_size,
                                               iou_threshold,
                                               score_threshold)  # 一维索引
        # 映射索引
        idx = tf.gather(idx, indices)  # [m,1]
        # padding值为 -1
        pad_num = tf.maximum(0, max_output_size - tf.shape(idx)[0])
        return tf.pad(idx, paddings=[[0, pad_num], [0, 0]], mode='constant', constant_values=-1)

    # 经过类别nms后保留的class_id 索引
    nms_keep = tf.map_fn(fn=per_class_nms, elems=unique_class_ids)  # (s,max_output_size,1)
    # 打平
    nms_keep = tf.reshape(nms_keep, shape=[-1])  # [s]
    # 去除padding
    nms_keep = tf.gather_nd(nms_keep, tf.where(nms_keep > -1))  # [s]

    # 获取类别nms的边框,评分,类别以及logits
    output_boxes = tf.gather(keep_boxes, nms_keep)
    output_scores = tf.gather(keep_class_scores, nms_keep)
    output_class_ids = tf.gather(keep_class_ids, nms_keep)
    output_class_logits = tf.gather(keep_class_logits, nms_keep)

    # 保留评分最高的top N
    top_num = tf.minimum(max_output_size, tf.shape(output_scores)[0])
    top_idx = tf.nn.top_k(output_scores, k=top_num)[1]  # top_k返回tuple(values,indices)
    output_boxes = tf.gather(output_boxes, top_idx)
    output_scores = tf.gather(output_scores, top_idx)
    output_class_ids = tf.gather(output_class_ids, top_idx)
    output_class_logits = tf.gather(output_class_logits, top_idx)

    # 增加padding,返回最终结果
    return [pad_to_fixed_size(output_boxes, max_output_size),
            pad_to_fixed_size(tf.expand_dims(output_scores, axis=1), max_output_size),
            pad_to_fixed_size(tf.expand_dims(output_class_ids, axis=1), max_output_size),
            pad_to_fixed_size(output_class_logits, max_output_size)]


class ProposalToDetectBox(keras.layers.Layer):
    """
    根据候选框生成最终的检测框
    """

    def __init__(self, score_threshold=0.7, output_box_num=100, iou_threshold=0.3, **kwargs):
        """
        :param score_threshold: 分数阈值
        :param output_box_num: 生成proposal 边框数量
        :param iou_threshold: nms iou阈值; 由于是类别相关的iou值较低
        """
        self.score_threshold = score_threshold
        self.output_box_num = output_box_num
        self.iou_threshold = iou_threshold
        super(ProposalToDetectBox, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        应用边框回归，并使用nms生成最后的边框
        :param inputs:
        inputs[0]: deltas, [batch_size,N,(dy,dx,dh,dw)]   N是proposal数量;
        inputs[1]: class logits [batch_size,N,num_classes]
        inputs[2]: proposals [batch_size,N,(y1,x1,y2,x2,tag)]
        :param kwargs:
        :return:
        """
        deltas = inputs[0]
        class_logits = inputs[1]
        proposals = inputs[2][..., :-1]  # 去除tag列

        # 应用边框回归

        boxes = tf.map_fn(lambda x: apply_regress(*x),
                          elems=[deltas, proposals],
                          dtype=tf.float32)

        # # 非极大抑制
        options = {"max_output_size": self.output_box_num,
                   "iou_threshold": self.iou_threshold,
                   "score_threshold": self.score_threshold}

        outputs = tf.map_fn(lambda x: detect_boxes(*x, **options),
                            elems=[boxes, class_logits],
                            dtype=[tf.float32] * 2 + [tf.int64] + [tf.float32])
        return outputs

    def compute_output_shape(self, input_shape):
        """
        注意多输出，call返回值必须是列表
        :param input_shape:
        :return: [boxes,scores,class_ids,class_logits]
        """
        return [(input_shape[0][0], self.output_box_num, 4 + 1),
                (input_shape[0][0], self.output_box_num, 1 + 1),
                (input_shape[0][0], self.output_box_num, 1 + 1),
                (input_shape[0][0], self.output_box_num, input_shape[1][-1])]
