# -*- coding: utf-8 -*-
"""
Created on 2019/4/8 上午6:29
通用工具类

@author: mick.yi

"""


def log(text, array=None):
    """
    日志输出，如果是数组，增加形状，最大值，最小值
    :param text:
    :param array:
    :return:
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)
