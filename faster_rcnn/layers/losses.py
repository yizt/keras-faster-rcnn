# -*- coding: utf-8 -*-
"""
Created on 2018/12/4 10:46

@author: mick.yi
损失函数层
"""
import tensorflow as tf


def rpn_cls_loss(predict_cls_ids, true_cls_ids, indices):
    """

    :param predict_cls_ids: 预测的anchors类别，(batch_num,anchors_num,2)
    :param true_cls_ids:实际的anchors类别，(batch_num,rpn_train_anchors,2)
    :param indices: 正负样本索引，(batch_num,rpn_train_anchors,2)，指定索引位置，
                               和类别 1：正样本，0：负样本，-1 padding
    :return:
    """
    # 正负样本索引号
    indices = tf.where(tf.not_equal(indices[:, :, -1] , -1))

