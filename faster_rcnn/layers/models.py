# -*- coding: utf-8 -*-
"""
Created on 2018/12/15 下午10:35

@author: mick.yi

frcnn模型

"""
import re
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
from faster_rcnn.utils.parallel_model import ParallelModel
from faster_rcnn.utils.utils import log
from faster_rcnn.layers.base_net import conv_block_5d, identity_block_5d


def rpn_net(config, stage='train'):
    batch_size = config.IMAGES_PER_GPU
    input_image = Input(shape=config.IMAGE_INPUT_SHAPE)
    input_class_ids = Input(shape=(config.MAX_GT_INSTANCES, 1 + 1))
    input_boxes = Input(shape=(config.MAX_GT_INSTANCES, 4 + 1))
    input_image_meta = Input(shape=(12,))

    # 特征及预测结果
    features = resnet50(input_image)
    boxes_regress, class_logits = rpn(features, config.RPN_ANCHOR_NUM)

    # 生成anchor
    anchors, anchors_tag = Anchor(config.RPN_ANCHOR_HEIGHTS,
                                  config.RPN_ANCHOR_WIDTHS,
                                  config.RPN_ANCHOR_BASE_SIZE,
                                  config.RPN_ANCHOR_RATIOS,
                                  config.RPN_ANCHOR_SCALES,
                                  config.BACKBONE_STRIDE, name='gen_anchors')(features)
    # 裁剪到窗口内
    # anchors = UniqueClipBoxes(config.IMAGE_INPUT_SHAPE, name='clip_anchors')(anchors)
    # windows = Lambda(lambda x: x[:, 7:11])(input_image_meta)
    # anchors = ClipBoxes()([anchors, windows])

    if stage == 'train':
        # 生成分类和回归目标
        rpn_targets = RpnTarget(batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, name='rpn_target')(
            [input_boxes, input_class_ids, anchors, anchors_tag])  # [deltas,cls_ids,indices,..]
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
                                                      iou_threshold=config.RPN_NMS_THRESHOLD_INFERENCE,
                                                      name='rpn2proposals')(
            [boxes_regress, class_logits, anchors, anchors_tag])
        return Model(inputs=[input_image, input_image_meta],
                     outputs=[detect_boxes, class_scores])


def frcnn(config, stage='train'):
    batch_size = config.IMAGES_PER_GPU
    # 输入
    input_image = Input(shape=config.IMAGE_INPUT_SHAPE, name='input_image')
    input_image_meta = Input(shape=(12,), name='input_image_meta')
    gt_class_ids = Input(shape=(config.MAX_GT_INSTANCES, 1 + 1), name='input_gt_class_ids')
    gt_boxes = Input(shape=(config.MAX_GT_INSTANCES, 4 + 1), name='input_gt_boxes')

    # 特征及预测结果
    features = config.base_fn(input_image)
    boxes_regress, class_logits = rpn(features, config.RPN_ANCHOR_NUM)

    # 生成anchor
    anchors, anchors_tag = Anchor(config.RPN_ANCHOR_HEIGHTS,
                                  config.RPN_ANCHOR_WIDTHS,
                                  config.RPN_ANCHOR_BASE_SIZE,
                                  config.RPN_ANCHOR_RATIOS,
                                  config.RPN_ANCHOR_SCALES,
                                  config.BACKBONE_STRIDE, name='gen_anchors')(features)
    # 裁剪到输入形状内
    # anchors = UniqueClipBoxes(config.IMAGE_INPUT_SHAPE, name='clip_anchors')(anchors)
    windows = Lambda(lambda x: x[:, 7:11])(input_image_meta)
    # anchors = ClipBoxes()([anchors, windows])

    # 应用分类和回归生成proposal
    output_box_num = config.POST_NMS_ROIS_TRAINING if stage == 'train' else config.POST_NMS_ROIS_INFERENCE
    iou_threshold = config.RPN_NMS_THRESHOLD_TRAINING if stage == 'train' else config.RPN_NMS_THRESHOLD_INFERENCE
    proposal_boxes, _, _ = RpnToProposal(batch_size, output_box_num=output_box_num,
                                         iou_threshold=iou_threshold,
                                         name='rpn2proposals')([boxes_regress, class_logits, anchors, anchors_tag])
    # proposal裁剪到图像窗口内
    proposal_boxes_coordinate, proposal_boxes_tag = Lambda(lambda x: [x[..., :4], x[..., 4:]])(proposal_boxes)
    # proposal_boxes_coordinate = ClipBoxes()([proposal_boxes_coordinate, windows])
    proposal_boxes_coordinate = UniqueClipBoxes(config.IMAGE_INPUT_SHAPE,
                                                name='clip_proposals')(proposal_boxes_coordinate)
    # 最后再合并tag返回
    proposal_boxes = Lambda(lambda x: tf.concat(x, axis=-1))([proposal_boxes_coordinate, proposal_boxes_tag])

    if stage == 'train':
        # 生成分类和回归目标
        rpn_targets = RpnTarget(batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, name='rpn_target')(
            [gt_boxes, gt_class_ids, anchors, anchors_tag])  # [deltas,cls_ids,indices,..]
        rpn_deltas, rpn_cls_ids, anchor_indices = rpn_targets[:3]
        # 定义rpn损失layer
        cls_loss_rpn = Lambda(lambda x: rpn_cls_loss(*x), name='rpn_class_loss')(
            [class_logits, rpn_cls_ids, anchor_indices])
        regress_loss_rpn = Lambda(lambda x: rpn_regress_loss(*x), name='rpn_bbox_loss')(
            [boxes_regress, rpn_deltas, anchor_indices])

        # 检测网络的分类和回归目标
        roi_deltas, roi_class_ids, train_rois, rcnn_miss_gt_num, pos_roi_num = DetectTarget(batch_size,
                                                                                            config.TRAIN_ROIS_PER_IMAGE,
                                                                                            config.ROI_POSITIVE_RATIO,
                                                                                            name='rcnn_target')(
            [gt_boxes, gt_class_ids, proposal_boxes])
        # 检测网络
        rcnn_deltas, rcnn_class_logits = rcnn(features, train_rois, config.NUM_CLASSES, config.IMAGE_MAX_DIM,
                                              config.head_fn, pool_size=config.POOL_SIZE,
                                              fc_layers_size=config.RCNN_FC_LAYERS_SIZE)

        # 检测网络损失函数
        regress_loss_rcnn = Lambda(lambda x: detect_regress_loss(*x), name='rcnn_bbox_loss')(
            [rcnn_deltas, roi_deltas, roi_class_ids])
        cls_loss_rcnn = Lambda(lambda x: detect_cls_loss(*x), name='rcnn_class_loss')(
            [rcnn_class_logits, roi_class_ids])
        # 自定义度量命名
        gt_num, positive_num, negative_num, rpn_miss_gt_num, gt_match_min_iou = rpn_targets[3:]
        gt_num = Lambda(lambda x: tf.identity(x), name='identity_gt_num')(gt_num)
        positive_num = Lambda(lambda x: tf.identity(x), name='identity_positive_num')(positive_num)
        negative_num = Lambda(lambda x: tf.identity(x), name='identity_negative_num')(negative_num)
        rpn_miss_gt_num = Lambda(lambda x: tf.identity(x), name='identity_rpn_miss_gt_num')(rpn_miss_gt_num)
        gt_match_min_iou = Lambda(lambda x: tf.identity(x), name='identity_gt_match_min_iou')(gt_match_min_iou)
        rcnn_miss_gt_num = Lambda(lambda x: tf.identity(x), name='identity_rcnn_miss_gt_num')(rcnn_miss_gt_num)
        pos_roi_num = Lambda(lambda x: tf.identity(x), name='identity_pos_roi_num')(pos_roi_num)

        # 构建模型
        model = Model(inputs=[input_image, input_image_meta, gt_class_ids, gt_boxes],
                      outputs=[cls_loss_rpn, regress_loss_rpn, regress_loss_rcnn, cls_loss_rcnn] + [
                          gt_num, positive_num, negative_num, rpn_miss_gt_num, gt_match_min_iou,
                          pos_roi_num, rcnn_miss_gt_num])  # 在并行model中所有自定义度量必须在output中
        # 多gpu训练
        if config.GPU_COUNT > 1:
            model = ParallelModel(model, config.GPU_COUNT)
        return model
    else:  # 测试阶段
        # 检测网络
        rcnn_deltas, rcnn_class_logits = rcnn(features, proposal_boxes, config.NUM_CLASSES, config.IMAGE_MAX_DIM,
                                              config.head_fn, pool_size=config.POOL_SIZE,
                                              fc_layers_size=config.RCNN_FC_LAYERS_SIZE)
        # 处理类别相关
        rcnn_deltas = layers.Lambda(lambda x: deal_delta(*x), name='deal_delta')([rcnn_deltas, rcnn_class_logits])
        # 应用分类和回归生成最终检测框
        detect_boxes, class_scores, detect_class_ids, detect_class_logits = ProposalToDetectBox(
            score_threshold=config.DETECTION_MIN_CONFIDENCE,
            output_box_num=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD,
            name='proposals2detectboxes')(
            [rcnn_deltas, rcnn_class_logits, proposal_boxes])
        # 裁剪到窗口内部
        detect_boxes_coordinate, detect_boxes_tag = Lambda(lambda x: [x[..., :4], x[..., 4:]])(detect_boxes)
        detect_boxes_coordinate = ClipBoxes()([detect_boxes_coordinate, windows])
        # 最后再合并tag返回
        detect_boxes = Lambda(lambda x: tf.concat(x, axis=-1))([detect_boxes_coordinate, detect_boxes_tag])
        image_meta = Lambda(lambda x: tf.identity(x))(input_image_meta)  # 原样返回
        return Model(inputs=[input_image, input_image_meta],
                     outputs=[detect_boxes, class_scores, detect_class_ids, detect_class_logits, image_meta])


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


def rcnn(base_layers, rois, num_classes, image_max_dim, head_fn, pool_size=(7, 7), fc_layers_size=1024):
    # RoiAlign
    x = RoiAlign(image_max_dim, pool_size=pool_size)([base_layers, rois])  #
    # 收缩维度
    shared_layer = head_fn(x)
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


def set_trainable(layer_regex, keras_model, indent=0, verbose=1):
    """Sets model layers as trainable if their names match
    the given regular expression.
    """
    # Print message on the first call (but not on recursive calls)
    if verbose > 0 and keras_model is None:
        log("Selecting layers to train")

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
        else keras_model.layers

    for layer in layers:
        # Is the layer a model?
        if layer.__class__.__name__ == 'Model':
            print("In model: ", layer.name)
            set_trainable(
                layer_regex, keras_model=layer, indent=indent + 4)
            continue

        if not layer.weights:
            continue
        # Is it trainable?
        trainable = bool(re.fullmatch(layer_regex, layer.name))
        # Update layer. If layer is a container, update inner layer.
        if layer.__class__.__name__ == 'TimeDistributed':
            layer.layer.trainable = trainable
        else:
            layer.trainable = trainable
        # Print trainable layer names
        if trainable and verbose > 0:
            log("{}{:20}   ({})".format(" " * indent, layer.name,
                                        layer.__class__.__name__))


def main():
    # print(keras.backend.image_data_format())
    # model = resnet50(Input((224, 224, 3)))
    # model.summary()
    x = tf.ones(shape=(5, 4, 1, 1, 3))
    y = tf.squeeze(x, 3)
    sess = tf.Session()
    print(sess.run(y))


if __name__ == '__main__':
    main()
