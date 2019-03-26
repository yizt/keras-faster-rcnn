# -*- coding: utf-8 -*-
"""
   File Name：     inference
   Description :  frcnn预测
   Author :       mick.yi
   date：          2019/2/13
"""
import numpy as np
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from faster_rcnn.preprocess.input import VocDataset
from faster_rcnn.utils.image import load_image_gt
from faster_rcnn.config import current_config as config
from faster_rcnn.utils import visualize, np_utils
from faster_rcnn.layers import models


def class_map_to_id_map(class_mapping):
    id_map = {}
    for k, v in class_mapping.items():
        id_map[v] = k
    return id_map


def main():
    # 加载数据
    dataset = VocDataset(config.voc_path, class_mapping=config.CLASS_MAPPING)
    dataset.prepare()
    all_img_info = [info for info in dataset.get_image_info_list() if info['type'] == 'trainval']  # 测试集

    # 加载模型
    m = models.frcnn(config, stage='test')
    m.load_weights(config.rcnn_weights, by_name=True)
    m.summary()

    # class map 转为 id map
    id_mapping = class_map_to_id_map(config.CLASS_MAPPING)

    def _show_inference(id, ax=None):
        image, image_meta, _ = load_image_gt(id,
                                             all_img_info[id]['filepath'],
                                             config.IMAGE_MAX_DIM,
                                             all_img_info[id]['boxes'])
        boxes, scores, class_ids, class_logits = m.predict(
            [np.expand_dims(image, axis=0), np.expand_dims(image_meta, axis=0)])
        boxes = np_utils.remove_pad(boxes[0])
        scores = np_utils.remove_pad(scores[0])[:, 0]
        class_ids = np_utils.remove_pad(class_ids[0])[:, 0]
        visualize.display_instances(image, boxes[:5],
                                    class_ids[:5],
                                    id_mapping,
                                    scores=scores[:5],
                                    ax=ax)

    # 随机展示9张图像
    image_ids = np.random.choice(len(all_img_info), 9, replace=False)
    fig = plt.figure(figsize=(20, 20))
    for idx, image_id in enumerate(image_ids):
        ax = fig.add_subplot(3, 3, idx + 1)
        _show_inference(image_id, ax)
    fig.savefig('demo_images/inferece_examples.{}.png'.format(np.random.randint(10)))


if __name__ == '__main__':
    main()
