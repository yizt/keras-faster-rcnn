# -*- coding: utf-8 -*-
"""
Created on 2018/12/4 10:46

@author: mick.yi
损失函数层
"""
import tensorflow as tf


def rpn_cls_loss(predict_cls_ids, true_cls_ids, indices):
    """
    rpn分类损失
    :param predict_cls_ids: 预测的anchors类别，(batch_num,anchors_num,2)
    :param true_cls_ids:实际的anchors类别，(batch_num,rpn_train_anchors,2)
    :param indices: 正负样本索引，(batch_num,rpn_train_anchors,(idx,tag))，
             idx:指定anchor索引位置，tag 1：正样本，0：负样本，-1 padding
    :return:
    """
    # 正负样本索引号
    train_indices = tf.where(tf.not_equal(indices[:, :, -1], 1))
    train_anchors = tf.gather_nd(indices, train_indices)  # (train_num,(idx,tag))
    # batch索引
    batch_indices = train_indices[:, 0]  # 训练的第一维是batch索引
    # 每个训练anchor的索引
    train_anchor_indices = train_anchors[:, 0]  # 每个anchor的在所有anchors中的索引
    # 每个训练anchor的2维索引
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


def smooth_l1_loss(y_true, y_predict):
    """
    smooth L1损失函数；   0.5*x^2 if |x| <1 else |x|-0.5; x是 diff
    :param y_true:[N,4]
    :param y_predict:[N,4]
    :return:
    """
    abs_diff = tf.abs(y_true, y_predict)
    loss = tf.where(abs_diff < 1, 0.5 * tf.pow(abs_diff, 2), abs_diff - 0.5)
    return tf.reduce_sum(loss, axis=1)


def rpn_regress_loss(predict_deltas, deltas, indices):
    """

    :param predict_deltas: 预测的回归目标，(batch_num,anchors_num,4)
    :param deltas: 真实的回归目标，(batch_num,rpn_train_anchors,4)
    :param indices: 正负样本索引，(batch_num,rpn_train_anchors,(idx,tag))，
             idx:指定anchor索引位置，tag 1：正样本，0：负样本，-1 padding
    :return:
    """
    train_postive_indices = tf.where(tf.equal(indices[:, :, -1], 1))
    # 只有正样本做回归
    train_anchors = tf.gather_nd(indices, train_postive_indices)  # (positive_num,(idx,tag))
    # batch索引
    batch_indices = train_postive_indices[:, 0]
    # anchor索引
    true_postive_indices = train_anchors[:, 0]
    # 正样本anchor的2维索引
    train_indices_2d = tf.stack([batch_indices, true_postive_indices], axis=1)
    # 正样本anchor预测的回归类型
    predict_deltas = tf.gather_nd(predict_deltas, train_indices_2d)
    # 真实回归目标,打平前两维
    shape = tf.shape(deltas)
    deltas = tf.reshape(deltas, [tf.reduce_prod(shape[:2]), shape[-1]])

    # Smooth-L1
    return smooth_l1_loss(deltas, predict_deltas)
