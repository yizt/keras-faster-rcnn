# -*- coding: utf-8 -*-
"""
Created on 2019/3/29 下午9:59

GT 工具类

@author: mick.yi

"""
import numpy as np
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from faster_rcnn.config import current_config as config
from faster_rcnn.preprocess.input import VocDataset
from faster_rcnn.utils import image as image_utils


def iou_distance(box_a, box_b):
    """
    iou距离
    :param box_a: [h,w]
    :param box_b: [h,w]
    :return:
    """
    if len(np.shape(box_a)) == 1:
        ha, wa = box_a[0], box_a[1]
        hb, wb = box_b[0], box_b[1]
        overlap = min(ha, hb) * min(wa, wb)
        iou = overlap / (ha * wa + hb * wb - overlap)
    else:
        ha, wa = box_a[:, 0], box_a[:, 1]
        hb, wb = box_b[:, 0], box_b[:, 1]
        overlap = np.minimum(ha, hb) * np.minimum(wa, wb)
        iou = overlap / (ha * wa + hb * wb - overlap)
    return 1. - iou


def gt_boxes_cluster(gt_boxes, centers=5):
    """
    聚类gt boxes长宽
    :param gt_boxes: numpy数组 [n,(y1,x1,y2,x2)]
    :param centers: 聚类中心个数
    :return: 聚类后的高度和宽度
    """

    height = gt_boxes[:, 2] - gt_boxes[:, 0]
    width = gt_boxes[:, 3] - gt_boxes[:, 1]
    hw = np.stack([height, width], axis=1)
    # 保存长宽数据
    np.save('/tmp/gt_height_width.npy', hw)

    # Kmeans聚类
    metric = distance_metric(type_metric.USER_DEFINED, func=iou_distance)
    init_centers = hw[np.random.choice(len(hw), centers, replace=False)]
    m = kmeans(hw, init_centers, metric=metric)
    m.process()
    cluster_centers = np.array(m.get_centers())

    # 聚类
    height = np.array([round(h, 2) for h in cluster_centers[:, 0]])
    width = np.array([round(w, 2) for w in cluster_centers[:, 1]])
    # 排序输出结果
    sort_indices = np.argsort(height)
    height = height[sort_indices]
    width = width[sort_indices]

    return list(height), list(width)


def main():
    dataset = VocDataset(config.voc_path, class_mapping=config.CLASS_MAPPING)
    dataset.prepare()

    # 获取缩放图像后的gt_boxes
    gt_boxes_list = []
    for info in dataset.get_image_info_list():
        h, w, window, scale, padding = image_utils.resize_meta(info['height'],
                                                               info['width'],
                                                               config.IMAGE_MAX_DIM)
        boxes = image_utils.adjust_box(info['boxes'], padding, scale)
        gt_boxes_list.append(boxes)

    gt_boxes = np.concatenate(gt_boxes_list, axis=0)  # 合并
    # 对高度和宽度聚类
    h, w = gt_boxes_cluster(gt_boxes, 5)
    print("h:{} \nw:{}".format(h, w))
    print("ratio:{}".format([round(x[0] / x[1], 2) for x in zip(h, w)]))


if __name__ == '__main__':
    main()
