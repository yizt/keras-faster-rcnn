# -*- coding: utf-8 -*-
"""
Created on 2019/3/29 下午9:59

GT 工具类

@author: mick.yi

"""
import numpy as np
from sklearn.cluster import KMeans
import operator
from functools import reduce


def gt_boxes_cluster(gt_boxes_list, centers=5):
    """
    聚类gt boxes长宽
    :param gt_boxes_list: numpy数组 [n,(y1,x1,y2,x2)] 列表
    :param centers: 聚类中心个数
    :return: 聚类后的高度和宽度
    """
    gt_boxes = reduce(operator.add, gt_boxes_list)
    # Kmeans聚类
    height = gt_boxes[:, 2] - gt_boxes[:, 0]
    width = gt_boxes[:, 3] - gt_boxes[:, 1]
    hw = np.stack([height, width], axis=1)

    # 保存长宽数据
    np.save('/tmp/gt_height_width.npy', hw)
    # 聚类
    m = KMeans(n_clusters=centers).fit(hw)
    height = m.cluster_centers_[:, 0]
    width = m.cluster_centers_[:, 1]
    return list(height), list(width)


def main():
    x = [[1, 3], [3, 2, 4], [1]]
    y = reduce(operator.add, x)
    print(y)


if __name__ == '__main__':
    main()
