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
from faster_rcnn.utils.generator import Generator
from faster_rcnn.layers import models
from faster_rcnn.utils import model_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler


def set_gpu_growth(gpu_count):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(gpu_count)])
    cfg = tf.ConfigProto(allow_soft_placement=True)  # because no supported kernel for GPU devices is available
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    keras.backend.set_session(session)


# 层名匹配
layer_regex = {
    # 网络头
    "heads": r"base_features|(rcnn\_.*)|(rpn\_.*)",
    # 指定的阶段开始
    "3+": r"base_features|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(rcnn\_.*)|(rpn\_.*)",
    "4+": r"base_features|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(rcnn\_.*)|(rpn\_.*)",
    "5+": r"base_features|(res5.*)|(bn5.*)|(rcnn\_.*)|(rpn\_.*)|",
    # 所有层
    "all": ".*",
}


def get_call_back(stage):
    """
    定义call back
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/frcnn-' + stage + '.{epoch:03d}.h5',
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True,
                                 period=5)

    scheduler = LearningRateScheduler(lambda epoch:
                                      config.LEARNING_RATE if epoch < 80 else config.LEARNING_RATE / 10.)

    log = TensorBoard(log_dir='log')
    return [checkpoint, scheduler, log]


def main(args):
    set_gpu_growth(config.GPU_COUNT)
    dataset = VocDataset(config.voc_path, class_mapping=config.CLASS_MAPPING)
    dataset.prepare()
    train_img_info = [info for info in dataset.get_image_info_list() if info['type'] == 'trainval']  # 训练集
    print("train_img_info:{}".format(len(train_img_info)))
    test_img_info = [info for info in dataset.get_image_info_list() if info['type'] == 'test']  # 测试集
    print("test_img_info:{}".format(len(test_img_info)))
    m = models.frcnn(config, stage='train')
    # 加载预训练模型
    init_epochs = args.init_epochs
    if args.init_epochs > 0:
        m.load_weights('/tmp/frcnn-rcnn.{:03d}.h5'.format(init_epochs), by_name=True)
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
    val_gen = Generator(test_img_info,
                        config.IMAGE_INPUT_SHAPE,
                        config.MEAN_PIXEL,
                        config.BATCH_SIZE,
                        config.MAX_GT_INSTANCES)
    # 训练conv3 及以上
    models.set_trainable(layer_regex['3+'], m)
    loss_names = ["rpn_bbox_loss", "rpn_class_loss", "rcnn_bbox_loss", "rcnn_class_loss"]
    model_utils.compile(m, config.LEARNING_RATE, config.LEARNING_MOMENTUM,
                        config.GRADIENT_CLIP_NORM, config.WEIGHT_DECAY, loss_names, config.LOSS_WEIGHTS)
    m.summary()
    # 增加个性化度量
    metric_names = ['gt_num', 'positive_anchor_num', 'negative_anchor_num', 'rpn_miss_gt_num',
                    'gt_match_min_iou', 'positive_roi_num', 'rcnn_miss_gt_num']
    model_utils.add_metrics(m, metric_names, m.outputs[-7:])

    # 训练
    m.fit_generator(train_gen.gen(),
                    epochs=args.epochs,
                    steps_per_epoch=len(train_img_info) // config.BATCH_SIZE,
                    verbose=1,
                    initial_epoch=init_epochs,
                    validation_data=val_gen.gen(),
                    validation_steps=5,  # 小一点，不影响训练速度太多
                    use_multiprocessing=True,
                    callbacks=get_call_back('rcnn'))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--epochs", type=int, default=120, help="epochs")
    parse.add_argument("--init_epochs", type=int, default=0, help="init_epochs")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
