# -*- coding: utf-8 -*-
"""
   File Name：     np_utils
   Description :  numpy 工具类
   Author :       mick.yi
   date：          2019/1/31
"""
import numpy as np


def pad_to_fixed_size(input_np, fixed_size):
    """
    增加padding到固定尺寸,在第二维增加一个标志位,0-padding,1-非padding
    :param input_np: 二维数组
    :param fixed_size:
    :return:
    """
    shape = input_np.shape
    # 增加tag
    np_array = np.pad(input_np, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    # 增加padding
    pad_num = max(0, fixed_size - shape[0])
    return np.pad(np_array, ((0, pad_num), (0, 0)), mode='constant', constant_values=0)


def remove_pad(input_np):
    """
    去除padding
    :param input_np:
    :return:
    """
    pad_tag = input_np[:, -1]  # 最后一维是padding 标志，1-非padding
    real_size = int(np.sum(pad_tag))
    return input_np[:real_size, :-1]


def main():
    x = np.ones(shape=(3, 3))
    pad_x = pad_to_fixed_size(x,2)
    print("pad_x.shape:{}".format(pad_x.shape))

    remove_pad_x = remove_pad(pad_x)
    print("remove_pad_x.shape:{}".format(remove_pad_x.shape))


if __name__ == '__main__':
    main()
