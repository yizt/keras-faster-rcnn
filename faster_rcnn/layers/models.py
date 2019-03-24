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
from faster_rcnn.layers.specific_to_agnostic import deal_delta
from faster_rcnn.layers.detect_boxes import ProposalToDetectBox
from faster_rcnn.layers.clip_boxes import ClipBoxes, UniqueClipBoxes
from faster_rcnn.layers.base_net import resnet50


def rpn_net(image_shape, config, stage='train'):
    batch_size = config.IMAGES_PER_GPU
    # input_image = Input(shape=image_shape)
    # input_class_ids = Input(shape=(max_gt_num, 1 + 1))
    # input_boxes = Input(shape=(max_gt_num, 4 + 1))
    # input_image_meta = Input(shape=(12,))
    input_image = Input(batch_shape=(batch_size,) + image_shape)
    input_class_ids = Input(batch_shape=(batch_size, config.MAX_GT_INSTANCES, 1 + 1))
    input_boxes = Input(batch_shape=(batch_size, config.MAX_GT_INSTANCES, 4 + 1))
    input_image_meta = Input(batch_shape=(batch_size, 12))
    # 特征及预测结果
    features = resnet50(input_image)
    boxes_regress, class_logits = rpn(features, config.RPN_ANCHOR_NUM)

    # 生成anchor
    anchors = Anchor(config.RPN_ANCHOR_BASE_SIZE,
                     config.RPN_ANCHOR_RATIOS,
                     config.RPN_ANCHOR_SCALES,
                     config.BACKBONE_STRIDE, name='gen_anchors')(features)
    # 裁剪到窗口内
    anchors = UniqueClipBoxes(image_shape, name='clip_anchors')(anchors)
    # windows = Lambda(lambda x: x[:, 7:11])(input_image_meta)
    # anchors = ClipBoxes()([anchors, windows])

    if stage == 'train':
        # 生成分类和回归目标
        rpn_targets = RpnTarget(batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, name='rpn_target')(
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
        detect_boxes, class_scores, _ = RpnToProposal(batch_size,
                                                      output_box_num=config.POST_NMS_ROIS_INFERENCE,
                                                      iou_threshold=config.RPN_NMS_THRESHOLD,
                                                      name='rpn2proposals')(
            [boxes_regress, class_logits, anchors])
        return Model(inputs=[input_image, input_image_meta],
                     outputs=[detect_boxes, class_scores])


def frcnn(image_shape, batch_size, num_classes, max_gt_num, image_max_dim, train_rois_per_image, roi_positive_ratio,
          stage='train'):
    # input_image = Input(batch_shape=(batch_size,)+image_shape)
    # gt_class_ids = Input(batch_shape=(batch_size, max_gt_num, 1 + 1))
    # gt_boxes = Input(batch_shape=(batch_size, max_gt_num, 4 + 1))
    # input_image_meta = Input(batch_shape=(batch_size, 12))
    input_image = Input(shape=image_shape)
    gt_class_ids = Input(shape=(max_gt_num, 1 + 1))
    gt_boxes = Input(shape=(max_gt_num, 4 + 1))
    input_image_meta = Input(shape=(12,))
    # 特征及预测结果
    features = resnet50(input_image)
    boxes_regress, class_logits = rpn(features, 9)

    # 生成anchor
    anchors = Anchor(128, [1, 2, 1 / 2], [1, 2 ** 1, 2 ** 2],
                     16, name='gen_anchors')(features)
    # 裁剪到输入形状内
    # anchors = UniqueClipBoxes(image_shape, name='clip_anchors')(anchors)
    windows = Lambda(lambda x: x[:, 7:11])(input_image_meta)
    anchors = ClipBoxes()([anchors, windows])

    # 应用分类和回归生成proposal
    proposal_boxes, _, _ = RpnToProposal(batch_size, output_box_num=2000 if stage == 'train' else 1000,
                                         iou_threshold=0.8,
                                         name='rpn2proposals')([boxes_regress, class_logits, anchors])
    # proposal裁剪到图像窗口内
    proposal_boxes_coordinate, proposal_boxes_tag = Lambda(lambda x: [x[..., :4], x[..., 4:]])(proposal_boxes)
    # windows = Lambda(lambda x: x[:, 7:11])(input_image_meta)
    proposal_boxes_coordinate = ClipBoxes()([proposal_boxes_coordinate, windows])
    # 最后再合并tag返回
    proposal_boxes = Lambda(lambda x: tf.concat(x, axis=-1))([proposal_boxes_coordinate, proposal_boxes_tag])

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
        # 检测网络
        rcnn_deltas, rcnn_class_logits = rcnn(features, proposal_boxes, num_classes, image_max_dim, pool_size=(7, 7),
                                              fc_layers_size=1024)
        # 处理类别相关
        rcnn_deltas = layers.Lambda(lambda x: deal_delta(*x), name='deal_delta')([rcnn_deltas, rcnn_class_logits])
        # 应用分类和回归生成最终检测框
        detect_boxes, class_scores, detect_class_ids, detect_class_logits = ProposalToDetectBox(
            score_threshold=0.05,
            output_box_num=100,
            name='proposals2detectboxes')(
            [rcnn_deltas, rcnn_class_logits, proposal_boxes])
        # 裁剪到窗口内部
        detect_boxes_coordinate, detect_boxes_tag = Lambda(lambda x: [x[..., :4], x[..., 4:]])(detect_boxes)
        detect_boxes_coordinate = ClipBoxes()([detect_boxes_coordinate, windows])
        # 最后再合并tag返回
        detect_boxes = Lambda(lambda x: tf.concat(x, axis=-1))([detect_boxes_coordinate, detect_boxes_tag])

        return Model(inputs=[input_image, input_image_meta],
                     outputs=[detect_boxes, class_scores, detect_class_ids, detect_class_logits])


def compile(keras_model, config, loss_names=[]):
    """
    编译模型，增加损失函数，L2正则化以
    :param keras_model:
    :param config:
    :param loss_names: 损失函数列表
    :return:
    """
    # 优化目标
    optimizer = keras.optimizers.SGD(
        lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM,
        clipnorm=config.GRADIENT_CLIP_NORM)
    # 增加损失函数，首先清除之前的，防止重复
    keras_model._losses = []
    keras_model._per_input_losses = {}

    for name in loss_names:
        layer = keras_model.get_layer(name)
        if layer is None or layer.output in keras_model.losses:
            continue
        loss = (tf.reduce_mean(layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.add_loss(loss)

    # 增加L2正则化
    # 跳过批标准化层的 gamma 和 beta 权重
    reg_losses = [
        keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
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
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.metrics_tensors.append(loss)


def add_metrics(keras_model, metric_name_list, metric_tensor_list):
    """
    增加度量
    :param keras_model: 模型
    :param metric_name_list: 度量名称列表
    :param metric_tensor_list: 度量张量列表
    :return: 无
    """
    for name, tensor in zip(metric_name_list, metric_tensor_list):
        keras_model.metrics_names.append(name)
        keras_model.metrics_tensors.append(tf.reduce_mean(tensor, keepdims=True))


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
    x = TimeDistributed(Conv2D(fc_layers_size, pool_size, padding='valid'), name='rcnn_fc1')(
        x)  # 变为(batch_size,roi_num,1,1,channels)
    x = TimeDistributed(layers.BatchNormalization(), name='rcnn_class_bn1')(x)
    x = layers.Activation(activation='relu')(x)

    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1), padding='valid'), name='rcnn_fc2')(x)
    x = TimeDistributed(layers.BatchNormalization(), name='rcnn_class_bn2')(x)
    x = layers.Activation(activation='relu')(x)

    # 收缩维度
    shared_layer = layers.Lambda(lambda a: tf.squeeze(tf.squeeze(a, 3), 2))(x)  # 变为(batch_size,roi_num,channels)

    # 分类
    class_logits = TimeDistributed(layers.Dense(num_classes, activation='linear'), name='rcnn_class_logits')(
        shared_layer)

    # 回归(类别相关)
    deltas = TimeDistributed(layers.Dense(4 * num_classes, activation='linear'), name='rcnn_deltas')(
        shared_layer)  # shape (batch_size,roi_num,4*num_classes)

    # 变为(batch_size,roi_num,num_classes,4)
    roi_num = backend.int_shape(deltas)[1]
    deltas = layers.Reshape((roi_num, num_classes, 4))(deltas)

    return deltas, class_logits


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
