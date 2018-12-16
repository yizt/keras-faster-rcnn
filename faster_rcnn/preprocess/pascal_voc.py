# -*- coding: utf-8 -*-
"""
Created on 2018/12/1 下午3:03

@author: mick.yi

voc数据集

"""
from six import raise_from
import os
from faster_rcnn.preprocess.input import Dataset

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def _find_node(parent, name, debug_name=None, parse=None):
    """
    查找xml中的节点
    :param parent: 父节点
    :param name: 待查找名称
    :param debug_name:
    :param parse: 节点数据类型
    :return:
    """
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


def get_voc_data(input_path):
    all_imgs = []

    classes_count = {}

    class_mapping = {}

    data_paths = [os.path.join(input_path, s) for s in ['VOC2007']]

    print('Parsing annotation files')

    for data_path in data_paths:

        annot_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')
        imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'train.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'val.txt')

        trainval_files = []
        test_files = []
        try:
            with open(imgsets_path_trainval) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.jpg')
        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip() + '.jpg')
        except Exception as e:
            if data_path[-7:] == 'VOC2012':
                # this is expected, most pascal voc distibutions dont have the test.txt file
                pass
            else:
                print(e)

        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
        idx = 0
        for annot in annots:
            try:
                idx += 1

                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {'filepath': os.path.join(imgs_path, element_filename),
                                       'width': element_width,
                                       'height': element_height, 'bboxes': []}

                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'

                for element_obj in element_objs:
                    class_name = element_obj.find('name').text
                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1

                    if class_name not in class_mapping:
                        # 类别id从1开始，0保留为背景
                        class_mapping[class_name] = len(class_mapping) + 1

                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    difficulty = int(element_obj.find('difficult').text) == 1
                    annotation_data['bboxes'].append(
                        {'class_name': class_name,
                         'class_id': class_mapping[class_name],
                         'x1': x1, 'x2': x2,
                         'y1': y1, 'y2': y2,
                         'difficult': difficulty})
                all_imgs.append(annotation_data)

            except Exception as e:
                print(e)
                continue
    return all_imgs, classes_count, class_mapping


class PascalVoc(Dataset):

    def load_voc(self, data_dir):
        all_imgs, _, class_map = get_voc_data(data_dir)

        # Add classes
        for class_name in class_map.keys():
            self.add_class("voc", class_map[class_name], class_name)

        # Add images
        for i, img_info in enumerate(all_imgs):
            self.add_image(
                "voc", image_id=i,
                path=img_info['filepath'],
                width=img_info["width"],
                height=img_info["height"],
                annotations=img_info['bboxes'])


if __name__ == '__main__':
    voc_path = '/Users/yizuotian/dataset/VOCdevkit/'
    get_voc_data(voc_path)
