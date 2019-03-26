# -*- coding: utf-8 -*-
"""
   File Name：     evaluate
   Description :   frcnn评估
   Author :       mick.yi
   date：          2019/3/4
"""
import argparse
import sys
import numpy as np
from faster_rcnn.preprocess.input import VocDataset
from faster_rcnn.config import current_config as config
from faster_rcnn.utils import np_utils, image as image_utils, eval_utils
from faster_rcnn.layers import models


def main(args):
    # 加载数据集
    dataset = VocDataset(config.voc_path, class_mapping=config.CLASS_MAPPING)
    dataset.prepare()
    print("len:{}".format(len(dataset.get_image_info_list())))
    test_image_info_list = [info for info in dataset.get_image_info_list() if info['type'] == 'test']
    print("len:{}".format(len(test_image_info_list)))
    # 加载模型
    m = models.frcnn(config, stage='test')
    if args.weight_path is not None:
        m.load_weights(args.weight_path, by_name=True)
    else:
        m.load_weights(config.rcnn_weights, by_name=True)
    m.summary()
    # 预测边框、得分、类别
    predict_boxes = []
    predict_scores = []
    predict_labels = []
    # 逐个图像处理
    for id in range(len(test_image_info_list)):
        image, image_meta, _ = image_utils.load_image_gt(id,
                                                         test_image_info_list[id]['filepath'],
                                                         config.IMAGE_MAX_DIM,
                                                         test_image_info_list[id]['boxes'])
        boxes, scores, class_ids, class_logits = m.predict(
            [np.expand_dims(image, axis=0), np.expand_dims(image_meta, axis=0)])
        boxes = np_utils.remove_pad(boxes[0])
        scores = np_utils.remove_pad(scores[0])[:, 0]
        class_ids = np_utils.remove_pad(class_ids[0])[:, 0]
        # 还原检测边框到
        window = image_meta[7:11]
        scale = image_meta[11]
        boxes = image_utils.recover_detect_boxes(boxes, window, scale)
        # 添加到列表中
        predict_boxes.append(boxes)
        predict_scores.append(scores)
        predict_labels.append(class_ids)
        if id % 100 == 0:
            print('预测完成：{}'.format(id + 1))

    # 以下是评估过程
    annotations = eval_utils.get_annotations(test_image_info_list, config.NUM_CLASSES)
    detections = eval_utils.get_detections(predict_boxes, predict_scores, predict_labels, config.NUM_CLASSES)
    average_precisions = eval_utils.voc_eval(annotations, detections, iou_threshold=0.5, use_07_metric=True)
    print("ap:{}".format(average_precisions))
    # 求mean ap 去除背景类
    mAP = np.mean(np.array(list(average_precisions.values()))[1:])
    print("mAP:{}".format(mAP))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
