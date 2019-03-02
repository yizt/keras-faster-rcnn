# -*- coding: utf-8 -*-
"""
   File Name：     eval_utils
   Description :  评估工具类
   Author :       mick.yi
   date：          2019/3/2
"""
import numpy as np
from . import np_utils


def get_detections(boxes, scores, predict_labels, num_classes, score_shreshold=0.05, max_boxes_num=100):
    """
    获取检测信息
    :param boxes: 检测边框，numpy数组 [num_images,N,(y1,x1,y2,x2,tag)]，tag=0为padding
    :param scores: 预测得分，numpy数组 [num_images,N,(score,tag)]，tag=0为padding
    :param predict_labels: 预测类别，numpy数组 [num_images,N,(predict_label,tag)]，tag=0为padding
    :param num_classes: 类别数
    :param score_shreshold: 评分阈值
    :param max_boxes_num:
    :return: list of list of numpy(num_boxes,5)  [num_images,[num_classes,[num_boxes,(y1,x1,y2,x2,scores)]]]
             每张图像，每个类别的预测边框；注意num_boxes是变化的；
    """
    # 初始化结果
    num_images = boxes.shape[0]
    all_dections = [[None for j in range(num_classes)] for i in range(num_images)]  # (num_images,num_classes)

    # 逐个图像处理
    for image_idx in range(num_images):
        # 去除padding
        cur_boxes = np_utils.remove_pad(boxes[image_idx])  # (n,4)
        cur_scores = np_utils.remove_pad(scores[image_idx])[:, 0]  # (n,)
        cur_predict_labels = np_utils.remove_pad(predict_labels[image_idx])[:, 0]  # (n,)
        # 过滤排序
        indices = np.where(cur_scores >= score_shreshold)[0]  # 选中的索引号，tuple的第一个值，一个一维numpy数组
        select_scores = cur_scores[indices]
        scores_sort_indices = np.argsort(select_scores * -1)[:max_boxes_num]  # (m,)选中评分排序过滤后的索引号

        # 最终的选中边框的索引号
        indices = indices[scores_sort_indices]
        # 选中的边框，得分，类别
        cur_boxes = cur_boxes[indices]
        cur_scores = cur_scores[indices]
        cur_predict_labels = cur_predict_labels[indices]
        # 合并边框和得分
        cur_detections = np.concatenate([cur_boxes, np.expand_dims(cur_scores, axis=1)], axis=1)

        # 逐个类别处理
        for class_id in range(num_classes):
            all_dections[image_idx, class_id] = cur_detections[cur_predict_labels == class_id]

    return all_dections


def get_annotations(metas, num_classes):
    """
    获取所有的编著
    :param metas: list of meta, 元数据信息
                        meta['boxes'] 是(n,4)数组,
                        meta['labels'] 是(n,1)数组
    :param num_classes: 类别数
    :return: list of list of numpy(num_boxes,4)  [num_images,[num_classes,[num_gt,(y1,x1,y2,x2)]]]
             每张图像，每个类别的GT边框; 注意num_gt是变化的
    """
    num_images = len(metas)
    all_annotations = [[None for j in range(num_classes)] for i in range(num_images)]  # (num_images,num_classes)
    for image_idx in range(num_images):
        for class_id in range(num_classes):
            indices = np.where(metas[image_idx]['labels'] == class_id)
            all_annotations[image_idx, class_id] = metas[image_idx]['boxes'][indices]

    return all_annotations


def voc_ap(rec, prec, use_07_metric=False):
    """
    voc 精度计算，此方法来自：https://github.com/rbgirshick/py-faster-rcnn
    :param rec: 召回率，numpy数组(n,)
    :param prec: 精度，numpy数组(n,)
    :param use_07_metric: 是否使用VOC 07的11点法
    :return: ap
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(all_annotations, all_detections, iou_threshold=0.5, use_07_metric=False):
    """
    voc数据集评估
    :param all_annotations:list of list of numpy(num_boxes,4) [num_images,[num_classes,[num_gt,(y1,x1,y2,x2)]]]
             每张图像，每个类别的GT边框; 注意num_gt是变化的
    :param all_detections:list of list of numpy(num_boxes,5) [num_images,[num_classes,[num_boxes,(y1,x1,y2,x2,scores)]]]
             每张图像，每个类别的预测边框；注意num_boxes是变化的；
    :param iou_threshold: iou阈值
    :param use_07_metric:
    :return: ap numpy数组，(num_classes,)
    """
    num_classes = len(all_annotations[0])
    num_images = len(all_annotations)
    average_precisions = {}
    # 逐个类别计算ap
    for class_id in range(num_classes):
        true_positives = np.zeros((0,))
        false_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_gt_boxes = 0
        # 逐个图像处理
        for image_id in range(num_images):
            gt_boxes = all_annotations[image_id][class_id]  # (n,y1,x1,y2,x2)
            num_gt_boxes += gt_boxes.shape[0]  # gt个数

            detected_gt_boxes = []  # 已经检测匹配过的gt边框

            for detect_box in all_detections[image_id][class_id]:
                scores = np.append(scores, detect_box[4])
                # 如果没有GT 边框
                if gt_boxes.shape[0] == 0:
                    true_positives = np.append(true_positives, 0)
                    false_positives = np.append(false_positives, 1)
                    continue

                # 计算iou
                iou = np_utils.compute_iou(gt_boxes, np.expand_dims(detect_box[:4], axis=1))  # (n,1)
                max_iou = np.max(iou, axis=0)[0]  # 与GT边框的最大iou值
                argmax_iou = np.argmax(iou, axis=0)[0]  # 最大iou值对应的GT
                # 如果超过iou阈值,且之前没有检测框匹配
                if max_iou >= iou_threshold and argmax_iou not in detected_gt_boxes:
                    true_positives = np.append(true_positives, 1)
                    false_positives = np.append(false_positives, 0)
                    detected_gt_boxes.append(argmax_iou)
                else:
                    true_positives = np.append(true_positives, 0)
                    false_positives = np.append(false_positives, 1)

        # 每个类别按照得分排序
        indices = np.argsort(scores * -1)
        true_positives = true_positives[indices]
        false_positives = false_positives[indices]

        # 累加
        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)

        # 计算召回率和精度
        recall = true_positives / num_gt_boxes
        precision = true_positives / np.max(true_positives + false_positives, np.finfo(np.float64).eps)

        # 计算ap
        average_precisions[class_id] = voc_ap(recall, precision, use_07_metric=use_07_metric)

    return average_precisions
