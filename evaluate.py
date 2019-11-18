# -*- coding: utf-8 -*-
"""
   File Name：     evaluate
   Description :   frcnn评估
   Author :       mick.yi
   date：          2019/3/4
"""
import argparse
import sys
import time
import numpy as np
from faster_rcnn.preprocess.input import VocDataset
from faster_rcnn.config import current_config as config
from faster_rcnn.utils import np_utils, image as image_utils, eval_utils
from faster_rcnn.layers import models
from faster_rcnn.utils.generator import TestGenerator


def main(args):
    from tensorflow.python.keras import backend as K
    import tensorflow as tf

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    K.set_session(session)
    # 覆盖参数
    config.IMAGES_PER_GPU = 1
    config.GPU_COUNT = 1
    config.DETECTION_MIN_CONFIDENCE = 0.0  # 评估阶段保证有足够的边框输出

    # 加载数据集
    dataset = VocDataset(config.voc_path, class_mapping=config.CLASS_MAPPING)
    dataset.prepare()
    print("len:{}".format(len(dataset.get_image_info_list())))
    test_image_info_list = [info for info in dataset.get_image_info_list()
                            if info['type'] == args.data_set][:args.eval_num]
    print("len:{}".format(len(test_image_info_list)))
    gen = TestGenerator(test_image_info_list,
                        config.IMAGE_INPUT_SHAPE,
                        config.MEAN_PIXEL)
    # 加载模型
    m = models.frcnn(config, stage='test')
    if args.weight_path is not None:
        m.load_weights(args.weight_path, by_name=True)
    else:
        m.load_weights(config.rcnn_weights, by_name=True)
    # m.summary()
    # 预测边框、得分、类别
    s_time = time.time()
    boxes, scores, class_ids, _, image_metas = m.predict_generator(
        gen,
        steps=gen.size,
        verbose=1,
        workers=4,
        use_multiprocessing=False)
    print("预测 {} 张图像,耗时：{} 秒".format(len(test_image_info_list), time.time() - s_time))
    # 去除padding
    image_metas = image_utils.batch_parse_image_meta(image_metas)
    predict_scores = [np_utils.remove_pad(score)[:, 0] for score in scores]
    predict_labels = [np_utils.remove_pad(label)[:, 0] for label in class_ids]
    # 还原检测边框到
    predict_boxes = [image_utils.recover_detect_boxes(np_utils.remove_pad(box), window, scale)
                     for box, window, scale in zip(boxes, image_metas['window'], image_metas['scale'])]

    # 以下是评估过程
    annotations = eval_utils.get_annotations(test_image_info_list, config.NUM_CLASSES)
    detections = eval_utils.get_detections(predict_boxes, predict_scores, predict_labels, config.NUM_CLASSES)
    average_precisions = eval_utils.voc_eval(annotations, detections, iou_threshold=0.5, use_07_metric=True)
    print("ap:{}".format(average_precisions))
    # 求mean ap 去除背景类
    mAP = np.mean(np.array(list(average_precisions.values()))[1:])
    print("mAP:{}".format(mAP))
    print("整个评估过程耗时：{} 秒".format(time.time() - s_time))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--data_set", type=str, default='test', help="dataset to evaluate")
    parse.add_argument("--eval_num", type=int, default=1000000, help="number of images to evaluate")
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
