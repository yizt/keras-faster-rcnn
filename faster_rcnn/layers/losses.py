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
    :param indices: 正负样本索引，(batch_num,rpn_train_anchors,(indices,tag))，
             indices:指定anchor索引位置，tag 1：正样本，0：负样本，-1 padding
    :return:
    """
    # 正负样本索引号
    train_indices = tf.where(tf.not_equal(indices[:, :, -1], 1))
    train_anchors = tf.gather_nd(indices, train_indices)  # (train_num,(indices,tag))
    # batch索引
    batch_indices = train_indices[:, 0]  # 训练的第一维是batch索引
    # 每个训练anchor的索引
    train_anchor_indices = train_anchors[:, 0]  # 每个anchor的在所有anchors中的索引
    # 每个训练anchor的类别
    train_indices_2d = tf.stack([batch_indices, train_anchor_indices], axis=1)
    # 获取预测的anchors类别
    predict_cls_ids = tf.gather_nd(predict_cls_ids, train_indices_2d)  # (train_num,2)

    # 真实的类别，打平前两维，batch和anchors打平
    shape = tf.shape(true_cls_ids)
    true_cls_ids = tf.reshape(true_cls_ids, [tf.reduce_prod(shape[:2]), shape[2]])

    # 交叉熵损失函数
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=true_cls_ids, logits=predict_cls_ids)
    return losses
