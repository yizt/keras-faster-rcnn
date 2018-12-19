# -*- coding: utf-8 -*-
"""
Created on 2018/12/15 下午10:35

@author: mick.yi

frcnn模型

"""
import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Reshape
import tensorflow as tf

from keras_applications.resnet50 import identity_block, conv_block
from faster_rcnn.layers.anchors import Anchor
from faster_rcnn.layers.target import RpnTarget
from faster_rcnn.layers.losses import rpn_cls_loss, rpn_regress_loss


def rpn_net(image_shape, max_gt_num, batch_size):
    input_image = Input(shape=image_shape)
    input_class_ids = Input(shape=(max_gt_num, 2 + 1))
    input_bboxes = Input(shape=(max_gt_num, 4 + 1))
    input_image_meta = Input(shape=(12,))
    # 特征及预测结果
    features = resnet50(input_image)
    # features = resnet_test_net(input_image)
    boxes_regress, class_ids = rpn(features, 9)

    # 生成anchor和目标
    anchors = Anchor(batch_size, 64, [1, 2, 1 / 2], [1, 2 ** 1, 2 ** 2],
                     32)([features, input_image_meta])
    target = RpnTarget(batch_size, 256, name='rpn_target')(
        [input_bboxes, input_class_ids, anchors])  # [cls_ids,deltas,indices]

    # 定义损失layer
    cls_loss = Lambda(lambda x: rpn_cls_loss(*x), name='rpn_class_loss')(
        [class_ids, target[0], target[2]])
    regress_loss = Lambda(lambda x: rpn_regress_loss(*x), name='rpn_bbox_loss')(
        [boxes_regress, target[1], target[2]])

    return Model(inputs=[input_image, input_image_meta, input_class_ids, input_bboxes],
                 outputs=[cls_loss, regress_loss])


def compile(keras_model, config, learning_rate, momentum):
    """Gets the model ready for training. Adds losses, regularization, and
    metrics. Then calls the Keras compile() function.
    """
    # Optimizer object
    optimizer = keras.optimizers.SGD(
        lr=learning_rate, momentum=momentum,
        clipnorm=config.GRADIENT_CLIP_NORM)
    # Add Losses
    # First, clear previously set losses to avoid duplication
    keras_model._losses = []
    keras_model._per_input_losses = {}
    loss_names = ["rpn_bbox_loss", "rpn_class_loss"]  # , "rpn_bbox_loss",rpn_class_loss
    for name in loss_names:
        layer = keras_model.get_layer(name)
        if layer.output in keras_model.losses:
            continue
        loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.add_loss(loss)

    # Add L2 Regularization
    # Skip gamma and beta weights of batch normalization layers.
    reg_losses = [
        keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
        for w in keras_model.trainable_weights
        if 'gamma' not in w.name and 'beta' not in w.name]
    keras_model.add_loss(tf.add_n(reg_losses))

    # Compile
    keras_model.compile(
        optimizer=optimizer,
        loss=[None] * len(keras_model.outputs))

    # Add metrics for losses
    for name in loss_names:
        if name in keras_model.metrics_names:
            continue
        layer = keras_model.get_layer(name)
        keras_model.metrics_names.append(name)
        loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.metrics_tensors.append(loss)

    # 增加GT个数，正样本anchor数指标的统计
    layer = keras_model.get_layer('rpn_target')
    keras_model.metrics_names.append('gt_num')
    keras_model.metrics_tensors.append(layer.output[3])

    keras_model.metrics_names.append('positive_anchor_num')
    keras_model.metrics_tensors.append(layer.output[4])

    keras_model.metrics_names.append('miss_match_gt_num')
    keras_model.metrics_tensors.append(layer.output[5])


#
#
# def rpn(base_layers, num_anchors):
#     x = Conv2D(512, (3, 3), padding='same',
#                activation='relu', kernel_initializer='normal',
#                name='rpn_conv1')(base_layers)  
#     x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x) 
#     x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(
#         x)  
#     return [x_class, x_regr, base_layers]

def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal')(base_layers)
    x_class = Conv2D(num_anchors * 2, (1, 1), kernel_initializer='uniform', activation='linear')(x)
    x_class = Reshape((-1, 2))(x_class)
    x_regr = Conv2D(num_anchors * 4, (1, 1),
                    kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None))(x)
    x_regr = Reshape((-1, 4))(x_regr)
    return x_regr, x_class


def resnet50(input):
    # Determine proper input shape
    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # # 确定精调层
    # no_train_model = Model(inputs=img_input, outputs=x)
    # for l in no_train_model.layers:
    #     if isinstance(l, layers.BatchNormalization):
    #         l.trainable = True
    #     else:
    #         l.trainable = False

    # model = Model(input, x, name='resnet50')

    return x


def resnet_test_net(input):
    x = Conv2D(512, (1, 1), strides=(32, 32))(input)
    return x


if __name__ == '__main__':
    print(keras.backend.image_data_format())
    model = resnet50(Input((224, 224, 3)))
    model.summary()
    # from keras_applications.resnet50 import ResNet50
    # m= ResNet50(True,weights='',input_shape=(224,224,3))
