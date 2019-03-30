# -*- coding: utf-8 -*-
"""
Created on 2019/3/29 下午9:59

GT 工具类

@author: mick.yi

"""
import numpy as np
from sklearn.cluster import KMeans
from faster_rcnn.config import current_config as config
from faster_rcnn.preprocess.input import VocDataset
from faster_rcnn.utils import image as image_utils


def gt_boxes_cluster(gt_boxes, centers=5):
    """
    聚类gt boxes长宽
    :param gt_boxes: numpy数组 [n,(y1,x1,y2,x2)]
    :param centers: 聚类中心个数
    :return: 聚类后的高度和宽度
    """

    # Kmeans聚类
    height = gt_boxes[:, 2] - gt_boxes[:, 0]
    width = gt_boxes[:, 3] - gt_boxes[:, 1]
    hw = np.stack([height, width], axis=1)

    # 保存长宽数据
    np.save('/tmp/gt_height_width.npy', hw)
    # 聚类
    m = KMeans(n_clusters=centers).fit(hw)
    height = [round(h, 2) for h in m.cluster_centers_[:, 0]]
    width = [round(w, 2) for w in m.cluster_centers_[:, 1]]
    return height, width


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
    print("h:{} \n w:{}".format(h, w))
    print("ratio:{}".format([round(x[0] / x[1], 2) for x in zip(h, w)]))


if __name__ == '__main__':
    main()
