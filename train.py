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
import tensorflow.python.keras as keras
from faster_rcnn.config import current_config as config
from faster_rcnn.preprocess.input import VocDataset
from faster_rcnn.utils.generator import Generator
from faster_rcnn.layers import models
from faster_rcnn.utils import model_utils
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler


def set_gpu_growth():
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(gpu_count)])
    config.GPU_LIST = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    config.GPU_COUNT = len(config.GPU_LIST)
    config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
    cfg = tf.ConfigProto(allow_soft_placement=True)  # because no supported kernel for GPU devices is available
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    keras.backend.set_session(session)


def lr_schedule(epoch):
    if epoch < 20:
        return config.LEARNING_RATE
    elif epoch < 60:
        return config.LEARNING_RATE / 10.
    else:
        return 1e-4


def get_call_back():
    """
    定义call back
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/frcnn-' + config.BASE_NET_NAME + '.{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True,
                                 save_freq='epoch')

    scheduler = LearningRateScheduler(lr_schedule)

    log = TensorBoard(log_dir='log')
    return [checkpoint, scheduler, log]


def main(args):
    set_gpu_growth()
    dataset = VocDataset(config.voc_path, class_mapping=config.CLASS_MAPPING)
    dataset.prepare()
    train_img_info = [info for info in dataset.get_image_info_list() if info['type'] == 'trainval']  # 训练集
    print("train_img_info:{}".format(len(train_img_info)))
    test_img_info = [info for info in dataset.get_image_info_list() if info['type'] == 'test']  # 测试集
    print("test_img_info:{}".format(len(test_img_info)))

    if config.GPU_COUNT > 1:
        with tf.device('/cpu:0'):
            m = models.frcnn(config, stage='train')
        m = keras.utils.multi_gpu_model(m, gpus=config.GPU_COUNT)
    else:
        m = models.frcnn(config, stage='train')

    # 加载预训练模型
    init_epochs = args.init_epochs
    if args.init_epochs > 0:
        m.load_weights('/tmp/frcnn-{}.{:03d}.h5'.format(config.BASE_NET_NAME, init_epochs), by_name=True)
    else:
        m.load_weights(config.pretrained_weights, by_name=True)
    # 生成器
    train_gen = Generator(train_img_info,
                          config.IMAGE_INPUT_SHAPE,
                          config.MEAN_PIXEL,
                          config.BATCH_SIZE,
                          config.MAX_GT_INSTANCES,
                          horizontal_flip=config.USE_HORIZONTAL_FLIP,
                          random_crop=config.USE_RANDOM_CROP)
    # 生成器
    val_gen = Generator(test_img_info[:500],
                        config.IMAGE_INPUT_SHAPE,
                        config.MEAN_PIXEL,
                        config.BATCH_SIZE,
                        config.MAX_GT_INSTANCES)
    # 训练conv3 及以上
    models.set_trainable(config.TRAIN_LAYERS, m)
    loss_names = ["rpn_bbox_loss", "rpn_class_loss", "rcnn_bbox_loss", "rcnn_class_loss"]
    model_utils.compile(m, config.LEARNING_RATE, config.LEARNING_MOMENTUM,
                        config.GRADIENT_CLIP_NORM, config.WEIGHT_DECAY, loss_names, config.LOSS_WEIGHTS)
    m.summary()
    # 增加个性化度量
    metric_names = ['gt_num', 'positive_anchor_num', 'negative_anchor_num', 'rpn_miss_gt_num',
                    'rpn_gt_min_max_iou', 'roi_num', 'positive_roi_num', 'negative_roi_num',
                    'rcnn_miss_gt_num', 'rcnn_miss_gt_num_as', 'gt_min_max_iou']
    model_utils.add_metrics(m, metric_names, m.outputs[-11:])

    # 训练
    m.fit_generator(train_gen,
                    epochs=args.epochs,
                    steps_per_epoch=len(train_gen),
                    verbose=1,
                    initial_epoch=init_epochs,
                    validation_data=val_gen,
                    validation_steps=len(val_gen),
                    use_multiprocessing=False,
                    workers=10,
                    callbacks=get_call_back())


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--epochs", type=int, default=80, help="epochs")
    parse.add_argument("--init_epochs", type=int, default=0, help="init_epochs")
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
