# -*- coding: utf-8 -*-
"""
   File Name：     roi_align
   Description :   RoiAlign层
   Author :       mick.yi
   date：          2019/2/1
"""
from keras import layers
import tensorflow as tf


class RoiAlign(layers.Layer):
    """
    将proposal边框投影到最后一层feature map上，并池化为7*7
    """

    def __init__(self, image_max_dim, pool_size=(7, 7), **kwargs):
        self.pool_size = pool_size
        self.image_max_dim = image_max_dim
        super(RoiAlign, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        inputs[0]: feature maps  [batch_num,H,W,feature_channel_num]
        inputs[1]: rois   [batch_num,roi_num,(y1,x1,y2,x2,tag)] , 训练和测试时，roi_num不同
        :param kwargs:
        :return:
        """
        features = inputs[0]
        rois = inputs[1][..., :-1]  # 去除tag列
        # 坐标归一化
        rois /= tf.constant(self.image_max_dim, dtype=tf.float32)
        # 生成batch index
        batch_size, roi_num = tf.shape(rois)[0], tf.shape(rois)[1]
        batch_index = tf.expand_dims(tf.range(batch_size), axis=1)
        batch_index = tf.tile(batch_index, [1, roi_num])
        batch_index = tf.reshape(batch_index, [-1])  # 类型[0,0,0,..,1,1,1...]
        # roi打平为二维
        rois = tf.reshape(rois, [-1, 4])

        # 停止反向传播（注释此部分，引起错误：It is possible you are working with a resizeable TensorArray and
        # stop_gradients is not allowing the gradients to be written）
        # rois = tf.stop_gradient(rois)
        # batch_index = tf.stop_gradient(batch_index)
        # RoiAlign
        output = tf.image.crop_and_resize(image=features,
                                          boxes=rois,
                                          box_ind=batch_index,
                                          crop_size=self.pool_size)  # (batch_size*roi_num,h,w,channels)
        # 转为(batch_size,roi_num,h,w,channels)
        shape = tf.shape(output)
        output = tf.reshape(output, [batch_size, roi_num, shape[1], shape[2], shape[3]], name='roi_align_output')
        return output

    def compute_output_shape(self, input_shape):
        channel_num = input_shape[0][-1]  # feature通道数
        return input_shape[1][:2] + self.pool_size + (channel_num,)  # (batch_size,roi_num,h,w,channels)


def main():
    x = tf.expand_dims(tf.range(2), axis=1)
    y = tf.tile(x, [1, 3])
    sess = tf.Session()
    print(sess.run(tf.reshape(y, [-1])))
    print(sess.run(x))
    print(sess.run(x[:100]))


if __name__ == '__main__':
    main()
