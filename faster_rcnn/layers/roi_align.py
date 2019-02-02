# -*- coding: utf-8 -*-
"""
   File Name：     roi_align
   Description :   RoiAlign层
   Author :       mick.yi
   date：          2019/2/1
"""
from keras import layers


class RoiAlign(layers.Layer):
    """
    将proposal边框投影到最后一层feature map上，并池化为7*7
    """

    def __init__(self, pool_size=(7, 7), max_proposal_num=2000, **kwargs):
        self.pool_size = pool_size
        self.max_proposal_num = max_proposal_num
        super(RoiAlign, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
