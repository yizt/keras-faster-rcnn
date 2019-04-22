# -*- coding: utf-8 -*-
"""
Created on 2019/04/12 下午9:42

@author: mick.yi

模型工具类

"""
import keras
import tensorflow as tf


def compile(keras_model, lr, momentum, clipnorm, weight_decay, loss_names=[], loss_weights={}):
    """
    编译模型，增加损失函数，L2正则化以
    :param keras_model:
    :param lr:
    :param momentum:
    :param clipnorm:
    :param weight_decay
    :param loss_names: 损失函数列表
    :param loss_weights:
    :return:
    """
    # 优化目标
    optimizer = keras.optimizers.SGD(
        lr=lr, momentum=momentum,
        clipnorm=clipnorm)
    # 增加损失函数，首先清除之前的，防止重复
    keras_model._losses = []
    keras_model._per_input_losses = {}

    for name in loss_names:
        layer = keras_model.get_layer(name)
        if layer is None or layer.output in keras_model.losses:
            continue
        loss = (tf.reduce_mean(layer.output, keepdims=True)
                * loss_weights.get(name, 1.))
        keras_model.add_loss(loss)

    # 增加L2正则化
    # 跳过批标准化层的 gamma 和 beta 权重
    reg_losses = [
        keras.regularizers.l2(weight_decay)(w) / tf.cast(tf.size(w), tf.float32)
        for w in keras_model.trainable_weights
        if 'gamma' not in w.name and 'beta' not in w.name]
    keras_model.add_loss(tf.add_n(reg_losses))

    # 编译
    keras_model.compile(
        optimizer=optimizer,
        loss=[None] * len(keras_model.outputs))  # 使用虚拟损失

    # 为每个损失函数增加度量
    for name in loss_names:
        if name in keras_model.metrics_names:
            continue
        layer = keras_model.get_layer(name)
        if layer is None:
            continue
        keras_model.metrics_names.append(name)
        loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * loss_weights.get(name, 1.))
        keras_model.metrics_tensors.append(loss)


def add_metrics(keras_model, metric_name_list, metric_tensor_list, gpu_num=1):
    """
    增加度量
    :param keras_model: 模型
    :param metric_name_list: 度量名称列表
    :param metric_tensor_list: 度量张量列表
    :param gpu_num: 数量
    :return: 无
    """
    for name, tensor in zip(metric_name_list, metric_tensor_list):
        keras_model.metrics_names.append(name)
        # 如果有gpu并行，且度量不是标量
        if gpu_num > 1:
            tensor = tf.cond(tf.greater(tf.size(tf.shape(tensor)), 0),
                             true_fn=lambda: tf.concat([tensor] * gpu_num, axis=0),
                             false_fn=lambda: tensor)
        keras_model.metrics_tensors.append(tf.reduce_mean(tensor, keepdims=True))
