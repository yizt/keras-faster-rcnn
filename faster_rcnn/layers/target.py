# -*- coding: utf-8 -*-
"""
Created on 2018/12/2 下午3:43

@author: mick.yi

分类回归目标层，包括rpn_target和detect_target

"""
import keras
import tensorflow as tf
from faster_rcnn.utils import tf_utils


def compute_iou(gt_boxes, anchors):
    """
    计算iou
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :param anchors: [M,(y1,x1,y2,x2)]
    :return: IoU [N,M]
    """
    gt_boxes = tf.expand_dims(gt_boxes, axis=1)  # [N,1,4]
    anchors = tf.expand_dims(anchors, axis=0)  # [1,M,4]
    # 交集
    intersect_w = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 3], anchors[:, :, 3]) -
                             tf.maximum(gt_boxes[:, :, 1], anchors[:, :, 1]))
    intersect_h = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 2], anchors[:, :, 2]) -
                             tf.maximum(gt_boxes[:, :, 0], anchors[:, :, 0]))
    intersect = intersect_h * intersect_w

    # 计算面积
    area_gt = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1]) * \
              (gt_boxes[:, :, 2] - gt_boxes[:, :, 0])
    area_anchor = (anchors[:, :, 3] - anchors[:, :, 1]) * \
                  (anchors[:, :, 2] - anchors[:, :, 0])

    # 计算并集
    union = area_gt + area_anchor - intersect
    # 交并比
    iou = intersect / union
    return iou


def regress_target(anchors, gt_boxes):
    """
    计算回归目标
    :param anchors: [N,(y1,x1,y2,x2)]
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :return: [N,(y1,x1,y2,x2)]
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 3] - gt_boxes[:, 1]
    # 中心点
    center_y = (anchors[:, 2] + anchors[:, 0]) * 0.5
    center_x = (anchors[:, 3] + anchors[:, 1]) * 0.5
    gt_center_y = (gt_boxes[:, 2] + gt_boxes[:, 0]) * 0.5
    gt_center_x = (gt_boxes[:, 3] + gt_boxes[:, 1]) * 0.5

    # 回归目标
    dy = (gt_center_y - center_y) / h
    dx = (gt_center_x - center_x) / w
    dh = tf.log(gt_h / h)
    dw = tf.log(gt_w / w)

    target = tf.stack([dy, dx, dh, dw], axis=1)
    return target


def rpn_targets_graph(gt_boxes, gt_cls, anchors, rpn_train_anchors=None):
    """
    处理单个图像的rpn分类和回归目标
    :param gt_boxes: GT 边框坐标 [MAX_GT_BOXs,(y1,x1,y2,x2,tag)] ,tag=-1 为padding
    :param gt_cls: GT 类别 [MAX_GT_BOXs,num_class+1] ;最后一位为tag, tag=-1 为padding
    :param anchors: [anchor_num,(y1,x1,y2,x2)]
    :param rpn_train_anchors: 训练样本数(256)
    :return:
    class_ids:[rpn_train_anchors,num_class]: anchor边框分类
    deltas:[rpn_train_anchors,(dy,dx,dh,dw)]：anchor边框回归目标
    indices:[rpn_train_anchors,(indices,tag)]: tag=1 为正样本，tag=0为负样本，tag=-1为padding
    """

    gt_indices = tf.where(tf.equal(gt_cls[:, -1], 1))
    # 获取真正的GT,去除标签位
    gt_boxes = tf.gather_nd(gt_boxes, gt_indices)[:, :-1]
    gt_cls = tf.gather_nd(gt_cls, gt_indices)[:, :-1]

    # 计算IoU
    iou = compute_iou(gt_boxes, anchors)
    #print("iou:{}".format(iou))

    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,

    # 每个anchors最大iou
    anchors_iou_max = tf.reduce_max(iou, axis=0)

    # 正样本索引号（iou>0.7),[[0],[2]]转为[0,2]
    positive_indices = tf.where(anchors_iou_max > 0.7)[:, 0]

    # 负样本索引号
    negative_indices = tf.where(anchors_iou_max < 0.3)[:, 0]

    # 正样本
    positive_num = int(rpn_train_anchors * 0.5)  # 正负比例1:1
    positive_num = tf.minimum(positive_num, tf.shape(positive_indices)[0])
    positive_indices = tf.random_shuffle(positive_indices)[:positive_num]
    positive_anchors = tf.gather_nd(anchors, positive_indices)

    # 负样本
    negative_num = tf.minimum(rpn_train_anchors - positive_num,
                              tf.shape(positive_indices)[0])
    negative_indices = tf.random_shuffle(negative_indices)[:negative_num]
    # negative_anchors = tf.gather_nd(anchors, negative_indices)

    # 找到正样本对应的GT boxes
    anchors_iou_argmax = tf.argmax(iou, axis=0)  # 每个anchor最大iou对应的GT 索引
    positive_gt_indices = tf.gather_nd(anchors_iou_argmax, positive_indices)
    positive_gt_boxes = tf.gather_nd(gt_boxes, positive_gt_indices)
    positive_gi_cls = tf.gather_nd(gt_cls, positive_gt_indices)

    # 回归目标
    deltas = regress_target(positive_anchors, positive_gt_boxes)

    # 计算padding
    pad_num = tf.maximum(0, rpn_train_anchors - positive_num - negative_num)

    class_ids = tf.pad(positive_gi_cls, [[0, negative_num + pad_num], [0, 0]])
    deltas = tf.pad(deltas, [[0, negative_num + pad_num], [0, 0]])

    # 处理索引,记录正负anchor索引位置，第二位为标志位1位前景、0为背景、-1位padding
    positive_part = tf.stack([positive_indices, tf.ones([positive_num], dtype=tf.int64)], axis=1)
    negative_part = tf.stack([negative_indices, tf.zeros([negative_num], dtype=tf.int64)], axis=1)
    pad_part = tf.ones([pad_num, 2], dtype=tf.int64) * -1
    indices = tf.concat([positive_part, negative_part, pad_part], axis=0)

    return class_ids, deltas, indices


class RpnTarget(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RpnTarget, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        计算分类和回归目标
        :param inputs:
        inputs[0]: GT 边框坐标 [batch_num,MAX_GT_BOXs,(y1,x1,y2,x2,tag)] ,tag=-1 为padding
        inputs[1]: GT 类别 [batch_num,MAX_GT_BOXs,num_class+1] ;最后一位为tag, tag=-1 为padding
        inputs[2]: Anchors [batch_num,anchor_num,(y1,x1,y2,x2)]
        :param kwargs:
        :return:
        """
        gt_boxes = inputs[0]
        gt_cls_ids = inputs[1]
        anchors = inputs[2]

        outputs = tf_utils.batch_slice(
            [gt_boxes, gt_cls_ids, anchors],
            lambda x, y, z: rpn_targets_graph(x, y, z, 256), 2)

        return outputs

    def compute_output_shape(self, input_shape):
        return [(None, 256, 2),  # 只有两类
                (None, 256, 4),
                (None, 256, 2)
                ]
