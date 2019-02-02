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
    :param predict_cls_ids: 预测的anchors类别，(batch_num,anchors_num,2) fg or bg
    :param true_cls_ids:实际的anchors类别，(batch_num,rpn_train_anchors,(class_id,tag))
             tag 1：正样本，0：负样本，-1 padding
    :param indices: 正负样本索引，(batch_num,rpn_train_anchors,(idx,tag))，
             idx:指定anchor索引位置，tag 1：正样本，0：负样本，-1 padding
    :return:
    """
    # 去除padding
    train_indices = tf.where(tf.not_equal(indices[:, :, -1], 0))  # 0为padding
    train_anchor_indices = tf.gather_nd(indices[..., 0], train_indices)  # 一维(batch*train_num,)，每个训练anchor的索引
    true_cls_ids = tf.gather_nd(true_cls_ids[..., 0], train_indices)  # 一维(batch*train_num,)
    # 转为onehot编码
    true_cls_ids = tf.where(true_cls_ids >= 1, tf.ones_like(true_cls_ids), tf.zeros_like(true_cls_ids))  # 前景类都为1
    true_cls_ids = tf.one_hot(true_cls_ids, depth=2)
    # batch索引
    batch_indices = train_indices[:, 0]  # 训练的第一维是batch索引
    # 每个训练anchor的2维索引
    train_indices_2d = tf.stack([batch_indices, tf.cast(train_anchor_indices, dtype=tf.int64)], axis=1)
    # 获取预测的anchors类别
    predict_cls_ids = tf.gather_nd(predict_cls_ids, train_indices_2d)  # (batch*train_num,2)

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
    abs_diff = tf.abs(y_true - y_predict, name='abs_diff')
    loss = tf.where(tf.less(abs_diff, 1), 0.5 * tf.pow(abs_diff, 2), abs_diff - 0.5)
    return tf.reduce_mean(loss, axis=1)


def rpn_regress_loss(predict_deltas, deltas, indices):
    """

    :param predict_deltas: 预测的回归目标，(batch_num, anchors_num, 4)
    :param deltas: 真实的回归目标，(batch_num, rpn_train_anchors, 4+1), 最后一位为tag, tag=0 为padding
    :param indices: 正负样本索引，(batch_num, rpn_train_anchors, (idx,tag))，
             idx:指定anchor索引位置，最后一位为tag, tag=0 为padding; 1为正样本，-1为负样本
    :return:
    """
    # 去除padding和负样本
    positive_indices = tf.where(tf.equal(indices[:, :, -1], 1))
    deltas = tf.gather_nd(deltas[..., :-1], positive_indices)  # (n,(dy,dx,dw,dh))
    true_positive_indices = tf.gather_nd(indices[..., 0], positive_indices)  # 一维，正anchor索引

    # batch索引
    batch_indices = positive_indices[:, 0]
    # 正样本anchor的2维索引
    train_indices_2d = tf.stack([batch_indices, tf.cast(true_positive_indices, dtype=tf.int64)], axis=1)
    # 正样本anchor预测的回归类型
    predict_deltas = tf.gather_nd(predict_deltas, train_indices_2d, name='rpn_regress_loss_predict_deltas')

    # Smooth-L1 # 非常重要，不然报NAN
    import keras.backend as K
    loss = K.switch(tf.size(deltas) > 0,
                    smooth_l1_loss(deltas, predict_deltas),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def main():
    x = tf.constant([1, 3, 6, 2, 3, 1, 0])
    x = tf.where(x >= 1, tf.ones_like(x), tf.zeros_like(x))
    # tf.scatter_update(x, tf.where(x > 1), 1)
    # x[x >= 1] = 1
    y = tf.one_hot(x, depth=2)
    sess = tf.Session()
    print(sess.run(y))


def detect_cls_loss(predict_cls_ids, true_cls_ids):
    """
    检测分类损失函数
    :param predict_cls_ids: 分类预测值 (batch_num, train_roi_num, num_classes)
    :param true_cls_ids: roi实际类别(batch_num, train_roi_num, (class_id,tag))， tag 为0 是padding
    :return:
    """
    # 去除padding
    indices = tf.where(tf.not_equal(true_cls_ids[..., -1], 0))
    predict_cls_ids = tf.gather_nd(predict_cls_ids, indices)  # 二维
    true_cls_ids = tf.gather_nd(true_cls_ids[..., 0], indices)  # 一维，类别id
    # 真实类别转one hot编码
    num_classes = tf.shape(predict_cls_ids)[1]
    true_cls_ids = tf.one_hot(true_cls_ids, depth=num_classes)

    # 交叉熵损失函数
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_cls_ids, logits=predict_cls_ids)
    return loss


def detect_regress_loss(predict_deltas, deltas):
    """
    检测网络回归损失
    :param predict_deltas: 回归预测值 (batch_num, train_roi_num, (dy,dx,dh,dw)
    :param deltas: 实际回归参数(batch_num, train_roi_num, (dy,dx,dh,dw,tag) ,tag：0-padding,-1-负样本,1-正样本
    :return:
    """
    # 去除padding和负样本，保留正样本
    indices = tf.where(tf.equal(deltas[..., -1], 1))
    predict_deltas = tf.gather_nd(predict_deltas, indices)
    deltas = tf.gather_nd(deltas[..., :-1], indices)

    # Smooth-L1 # 非常重要，不然报NAN
    import keras.backend as K
    loss = K.switch(tf.size(deltas) > 0,
                    smooth_l1_loss(deltas, predict_deltas),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


if __name__ == '__main__':
    main()
