# -*- coding: utf-8 -*-
"""
Created on 2018/12/16 上午9:30

@author: mick.yi

训练frcnn

"""

import argparse
import sys
import os
import tensorflow as tf
import keras
from faster_rcnn.config import current_config as config
from faster_rcnn.preprocess.input import VocDataset
from faster_rcnn.utils.generator import generator
from faster_rcnn.layers import models
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint


def set_gpu_growth():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    keras.backend.set_session(session)


def get_call_back(stage):
    """
    定义call back
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/frcnn-' + stage + '.{epoch:03d}.h5',
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=False,
                                 period=5)

    # 验证误差没有提升
    lr_reducer = ReduceLROnPlateau(monitor='loss',
                                   factor=0.1,
                                   cooldown=0,
                                   patience=10,
                                   min_lr=0)
    log = TensorBoard(log_dir='log')
    return [checkpoint, lr_reducer, log]


def main(args):
    set_gpu_growth()
    dataset = VocDataset(config.voc_path, class_mapping=config.CLASS_MAPPING)
    dataset.prepare()
    train_img_info = [info for info in dataset.get_image_info_list() if info['type'] == 'trainval']  # 训练集
    print("all_img_info:{}".format(len(train_img_info)))
    # 生成器
    gen = generator(train_img_info, config.IMAGES_PER_GPU, config.IMAGE_MAX_DIM, 50)
    #
    if 'rpn' in args.stages:
        m = models.rpn_net(config)
        m.load_weights(config.pretrained_weights, by_name=True)
        loss_names = ["rpn_bbox_loss", "rpn_class_loss"]
        models.compile(m, config, loss_names)
        # 增加个性化度量
        layer = m.get_layer('rpn_target')
        metric_names = ['gt_num', 'positive_anchor_num', 'miss_match_gt_num', 'gt_match_min_iou']
        models.add_metrics(m, metric_names, layer.output[-4:])
        m.summary()
        m.fit_generator(gen,
                        epochs=args.epochs,
                        steps_per_epoch=len(train_img_info) // config.IMAGES_PER_GPU,
                        verbose=1,
                        initial_epoch=args.init_epochs,
                        callbacks=get_call_back('rpn'))

        m.save(config.rpn_weights)
    if 'rcnn' in args.stages:
        m = models.frcnn((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3), config.BATCH_SIZE, config.NUM_CLASSES,
                         50, config.IMAGE_MAX_DIM, config.TRAIN_ROIS_PER_IMAGE, config.ROI_POSITIVE_RATIO)
        loss_names = ["rpn_bbox_loss", "rpn_class_loss", "rcnn_bbox_loss", "rcnn_class_loss"]
        models.compile(m, config, loss_names)
        # 增加个性化度量
        layer = m.get_layer('rpn_target')
        metric_names = ['gt_num', 'positive_anchor_num', 'miss_match_gt_num', 'gt_match_min_iou']
        models.add_metrics(m, metric_names, layer.output[-4:])

        layer = m.get_layer('rcnn_target')
        metric_names = ['rcnn_miss_match_gt_num']
        models.add_metrics(m, metric_names, layer.output[-1:])
        # 加载预训练模型
        if args.init_epochs > 0:
            m.load_weights(args.init_weight_path, by_name=True)
        elif os.path.exists(config.rpn_weights):  # 有rpn预训练模型就加载，没有直接加载resnet50预训练模型
            m.load_weights(config.rpn_weights, by_name=True)
        else:
            m.load_weights(config.pretrained_weights, by_name=True)
        m.summary()
        # 训练
        m.fit_generator(gen,
                        epochs=args.epochs,
                        steps_per_epoch=len(train_img_info) // config.IMAGES_PER_GPU,
                        verbose=1,
                        initial_epoch=args.init_epochs,
                        callbacks=get_call_back('rcnn'))

        if args.weight_path is not None:
            m.save(args.weight_path)
        else:
            m.save(config.rcnn_weights)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--stages", type=str, nargs='+', default=['rcnn'], help="stages: rpn、rcnn")
    parse.add_argument("--epochs", type=int, default=50, help="epochs")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--init_weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--init_epochs", type=int, default=0, help="weight path")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
