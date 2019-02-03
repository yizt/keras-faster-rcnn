# -*- coding: utf-8 -*-
"""
Created on 2018/12/15 下午10:35

@author: mick.yi

frcnn模型

"""
import keras
from keras import layers, backend
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Reshape, TimeDistributed
import tensorflow as tf
from faster_rcnn.layers.anchors import Anchor
from faster_rcnn.layers.target import RpnTarget, DetectTarget
from faster_rcnn.layers.proposals import RpnToProposal
from faster_rcnn.layers.roi_align import RoiAlign
from faster_rcnn.layers.losses import rpn_cls_loss, rpn_regress_loss, detect_regress_loss, detect_cls_loss


def rpn_net(image_shape, max_gt_num, batch_size, stage='train'):
    input_image = Input(shape=image_shape)
    input_class_ids = Input(shape=(max_gt_num, 1 + 1))
    input_boxes = Input(shape=(max_gt_num, 4 + 1))
    input_image_meta = Input(shape=(12,))
    # 特征及预测结果
    features = resnet50(input_image)
    # features = resnet_test_net(input_image)
    boxes_regress, class_logits = rpn(features, 9)

    # 生成anchor
    anchors = Anchor(batch_size, 64, [1, 2, 1 / 2], [1, 2 ** 1, 2 ** 2],
                     16, name='gen_anchors')([features, input_image_meta])

    if stage == 'train':
        # 生成分类和回归目标
        rpn_targets = RpnTarget(batch_size, 256, name='rpn_target')(
            [input_boxes, input_class_ids, anchors])  # [deltas,cls_ids,indices,..]
        deltas, cls_ids, anchor_indices = rpn_targets[:3]
        # 定义损失layer
        cls_loss = Lambda(lambda x: rpn_cls_loss(*x), name='rpn_class_loss')(
            [class_logits, cls_ids, anchor_indices])
        regress_loss = Lambda(lambda x: rpn_regress_loss(*x), name='rpn_bbox_loss')(
            [boxes_regress, deltas, anchor_indices])

        return Model(inputs=[input_image, input_image_meta, input_class_ids, input_boxes],
                     outputs=[cls_loss, regress_loss])
    else:  # 测试阶段
        # 应用分类和回归
        detect_boxes, class_scores = RpnToProposal(batch_size, output_box_num=10, name='rpn2proposals')(
            [boxes_regress, class_logits, anchors])
        return Model(inputs=[input_image, input_image_meta],
                     outputs=[detect_boxes, class_scores])


def frcnn(image_shape, batch_size, num_classes, max_gt_num, image_max_dim, train_rois_per_image, roi_positive_ratio,
          stage='train'):
    input_image = Input(shape=image_shape)
    gt_class_ids = Input(shape=(max_gt_num, 1 + 1))
    gt_boxes = Input(shape=(max_gt_num, 4 + 1))
    input_image_meta = Input(shape=(12,))
    # 特征及预测结果
    features = resnet50(input_image)
    # features = resnet_test_net(input_image)
    boxes_regress, class_logits = rpn(features, 9)

    # 生成anchor
    anchors = Anchor(batch_size, 64, [1, 2, 1 / 2], [1, 2 ** 1, 2 ** 2],
                     16, name='gen_anchors')([features, input_image_meta])

    if stage == 'train':
        # 生成分类和回归目标
        rpn_targets = RpnTarget(batch_size, 256, name='rpn_target')(
            [gt_boxes, gt_class_ids, anchors])  # [deltas,cls_ids,indices,..]
        rpn_deltas, rpn_cls_ids, anchor_indices = rpn_targets[:3]
        # 定义rpn损失layer
        cls_loss_rpn = Lambda(lambda x: rpn_cls_loss(*x), name='rpn_class_loss')(
            [class_logits, rpn_cls_ids, anchor_indices])
        regress_loss_rpn = Lambda(lambda x: rpn_regress_loss(*x), name='rpn_bbox_loss')(
            [boxes_regress, rpn_deltas, anchor_indices])

        # 应用分类和回归生成proposal
        proposal_boxes, _ = RpnToProposal(batch_size, output_box_num=1000, name='rpn2proposals')(
            [boxes_regress, class_logits, anchors])

        # 检测网络的分类和回归目标
        roi_deltas, roi_class_ids, train_rois, _ = DetectTarget(batch_size, train_rois_per_image, roi_positive_ratio,
                                                                name='rcnn_target')(
            [gt_boxes, gt_class_ids, proposal_boxes])
        # 检测网络
        rcnn_deltas, rcnn_class_logits = rcnn(features, train_rois, num_classes, image_max_dim, pool_size=(7, 7),
                                              fc_layers_size=1024)

        # 检测网络损失函数
        regress_loss_rcnn = Lambda(lambda x: detect_regress_loss(*x), name='rcnn_bbox_loss')(
            [rcnn_deltas, roi_deltas, roi_class_ids])
        cls_loss_rcnn = Lambda(lambda x: detect_cls_loss(*x), name='rcnn_class_loss')(
            [rcnn_class_logits, roi_class_ids])

        return Model(inputs=[input_image, input_image_meta, gt_class_ids, gt_boxes],
                     outputs=[cls_loss_rpn, regress_loss_rpn, regress_loss_rcnn, cls_loss_rcnn])
    else:  # 测试阶段
        # 应用分类和回归
        detect_boxes, class_scores = RpnToProposal(batch_size, output_box_num=10, name='rpn2proposals')(
            [boxes_regress, class_logits, anchors])
        return Model(inputs=[input_image, input_image_meta],
                     outputs=[detect_boxes, class_scores])


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
    loss_names = ["rpn_bbox_loss", "rpn_class_loss", "rcnn_bbox_loss",
                  "rcnn_class_loss"]  # , "rpn_bbox_loss",rpn_class_loss
    for name in loss_names:
        layer = keras_model.get_layer(name)
        if layer.output in keras_model.losses or layer is None:
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
    # 检测结果统计指标
    layer = keras_model.get_layer('rcnn_target')
    keras_model.metrics_names.append('rcnn_miss_match_gt_num')
    keras_model.metrics_tensors.append(layer.output[3])


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
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv')(
        base_layers)
    x_class = Conv2D(num_anchors * 2, (1, 1), kernel_initializer='uniform', activation='linear',
                     name='rpn_class_logits')(x)
    x_class = Reshape((-1, 2))(x_class)
    x_regr = Conv2D(num_anchors * 4, (1, 1),
                    kernel_initializer='normal', name='rpn_deltas')(x)
    x_regr = Reshape((-1, 4))(x_regr)
    return x_regr, x_class


def rcnn(base_layers, rois, num_classes, image_max_dim, pool_size=(7, 7), fc_layers_size=1024):
    x = RoiAlign(image_max_dim)([base_layers, rois])  #
    # 用卷积来实现两个全连接
    x = TimeDistributed(Conv2D(fc_layers_size, pool_size, padding='valid', name='rcnn_fc1'))(
        x)  # 变为(batch_size,roi_num,1,1,channels)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = layers.Activation(activation='relu')(x)

    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1), padding='valid', name='rcnn_fc2'))(x)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = layers.Activation(activation='relu')(x)

    # 收缩维度
    shared_layer = layers.Lambda(lambda a: tf.squeeze(tf.squeeze(a, 3), 2))(x)  # 变为(batch_size,roi_num,channels)

    # 分类
    class_logits = TimeDistributed(layers.Dense(num_classes, activation='linear', name='rcnn_class_logits'))(
        shared_layer)

    # 回归(类别相关)
    deltas = TimeDistributed(layers.Dense(4 * num_classes, activation='linear', name='rcnn_deltas'))(
        shared_layer)  # shape (batch_size,roi_num,4*num_classes)

    # 变为(batch_size,roi_num,num_classes,4)
    roi_num = backend.int_shape(deltas)[1]
    deltas = layers.Reshape((roi_num, num_classes, 4))(deltas)

    return deltas, class_logits


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


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

    # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

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


def main():
    # print(keras.backend.image_data_format())
    # model = resnet50(Input((224, 224, 3)))
    # model.summary()
    x = tf.ones(shape=(5, 4, 1, 1, 3))
    import keras.backend as K
    y = tf.squeeze(x, 3)
    sess = tf.Session()
    print(sess.run(y))


if __name__ == '__main__':
    main()
