# -*- coding: utf-8 -*-
"""
Created on 2018/11/13 10:11

@author: mick.yi

"""

import tensorflow as tf


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    # 行转列
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)
    # list转tensor
    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    # 如果返回单个值,不使用list
    if len(result) == 1:
        result = result[0]

    return result


def pad_to_fixed_size_with_negative(input_tensor, fixed_size, negative_num):
    # 输入尺寸
    input_size = tf.shape(input_tensor)[0]
    # tag 列 padding
    positive_num = input_size - negative_num  # 正例数
    # 正样本padding 1,负样本padding -1
    column_padding = tf.concat([tf.ones([positive_num]),
                                tf.ones([negative_num]) * -1],
                               axis=0)
    # 都转为float,拼接
    x = tf.concat([tf.cast(input_tensor, tf.float32), tf.expand_dims(column_padding, axis=1)], axis=1)
    # 不够的padding 0
    padding_size = tf.maximum(0, fixed_size - input_size)
    x = tf.pad(x, [[0, padding_size], [0, 0]], mode='CONSTANT', constant_values=0)
    return x


def pad_to_fixed_size(input_tensor, fixed_size):
    """
    增加padding到固定尺寸,在第二维增加一个标志位,0-padding,1-非padding
    :param input_tensor: 二维张量
    :param fixed_size:
    :param negative_num: 负样本数量
    :return:
    """
    input_size = tf.shape(input_tensor)[0]
    x = tf.pad(input_tensor, [[0, 0], [0, 1]], mode='CONSTANT', constant_values=1)
    # padding
    padding_size = tf.maximum(0, fixed_size - input_size)
    x = tf.pad(x, [[0, padding_size], [0, 0]], mode='CONSTANT', constant_values=0)
    return x


def pad_list_to_fixed_size(tensor_list, fixed_size):
    return [pad_to_fixed_size(tensor, fixed_size) for tensor in tensor_list]


def remove_pad(input_tensor):
    """

    :param input_tensor:
    :return:
    """
    pad_tag = input_tensor[..., -1]
    real_size = tf.cast(tf.reduce_sum(pad_tag), tf.int32)
    return input_tensor[:real_size, :-1]


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


def main():
    sess = tf.Session()
    x = sess.run(tf.maximum(3.0, 2.0))
    print(x)
    a = tf.ones(shape=(3, 3))
    b = pad_to_fixed_size(a, 4)
    c = remove_pad(b)
    print(sess.run(b))
    print(sess.run(c))


if __name__ == '__main__':
    main()
