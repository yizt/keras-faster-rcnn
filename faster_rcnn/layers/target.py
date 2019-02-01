# -*- coding: utf-8 -*-
"""
Created on 2018/12/2 下午3:43

@author: mick.yi

分类回归目标层，包括rpn_target和detect_target

"""
import keras
import tensorflow as tf
from faster_rcnn.utils import tf_utils


def compute_iou(gt_boxes, anchors):
    """
    计算iou
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :param anchors: [M,(y1,x1,y2,x2)]
    :return: IoU [N,M]
    """
    gt_boxes = tf.expand_dims(gt_boxes, axis=1)  # [N,1,4]
    anchors = tf.expand_dims(anchors, axis=0)  # [1,M,4]
    # 交集
    intersect_w = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 3], anchors[:, :, 3]) -
                             tf.maximum(gt_boxes[:, :, 1], anchors[:, :, 1]))
    intersect_h = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 2], anchors[:, :, 2]) -
                             tf.maximum(gt_boxes[:, :, 0], anchors[:, :, 0]))
    intersect = intersect_h * intersect_w

    # 计算面积
    area_gt = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1]) * \
              (gt_boxes[:, :, 2] - gt_boxes[:, :, 0])
    area_anchor = (anchors[:, :, 3] - anchors[:, :, 1]) * \
                  (anchors[:, :, 2] - anchors[:, :, 0])

    # 计算并集
    union = area_gt + area_anchor - intersect
    # 交并比
    iou = tf.divide(intersect, union, name='regress_target_iou')
    return iou


def regress_target(anchors, gt_boxes):
    """
    计算回归目标
    :param anchors: [N,(y1,x1,y2,x2)]
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :return: [N,(y1,x1,y2,x2)]
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 3] - gt_boxes[:, 1]
    # 中心点
    center_y = (anchors[:, 2] + anchors[:, 0]) * 0.5
    center_x = (anchors[:, 3] + anchors[:, 1]) * 0.5
    gt_center_y = (gt_boxes[:, 2] + gt_boxes[:, 0]) * 0.5
    gt_center_x = (gt_boxes[:, 3] + gt_boxes[:, 1]) * 0.5

    # 回归目标
    dy = (gt_center_y - center_y) / h
    dx = (gt_center_x - center_x) / w
    dh = tf.log(gt_h / h)
    dw = tf.log(gt_w / w)

    target = tf.stack([dy, dx, dh, dw], axis=1)
    target /= tf.constant([0.1, 0.1, 0.2, 0.2])
    # target = tf.where(tf.greater(target, 100.0), 100.0, target)
    return target


def rpn_targets_graph(gt_boxes, gt_cls, anchors, rpn_train_anchors=None):
    """
    处理单个图像的rpn分类和回归目标
    :param gt_boxes: GT 边框坐标 [MAX_GT_BOXs, (y1,x1,y2,x2,tag)] ,tag=-1 为padding
    :param gt_cls: GT 类别 [MAX_GT_BOXs, num_class+1] ;最后一位为tag, tag=-1 为padding
    :param anchors: [anchor_num, (y1,x1,y2,x2)]
    :param rpn_train_anchors: 训练样本数(256)
    :return:
    class_ids:[rpn_train_anchors,num_class]: anchor边框分类
    deltas:[rpn_train_anchors,(dy,dx,dh,dw)]：anchor边框回归目标
    indices:[rpn_train_anchors,(indices,tag)]: tag=1 为正样本，tag=0为负样本，tag=-1为padding
    """

    gt_indices = tf.where(tf.not_equal(gt_cls[:, -1], -1), name='rpn_target_gt_indices')
    # 获取真正的GT,去除标签位
    gt_boxes = tf.gather_nd(gt_boxes, gt_indices, name='rpn_target_gt_boxes')[:, :-1]
    gt_cls = tf.gather_nd(gt_cls, gt_indices, name='rpn_target_gt_cls')[:, :-1]

    # 计算IoU
    iou = compute_iou(gt_boxes, anchors)
    # print("iou:{}".format(iou))

    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,

    # 每个anchors最大iou
    anchors_iou_max = tf.reduce_max(iou, axis=0)

    # 正样本索引号（iou>0.7),[[0],[2]]转为[0,2]
    positive_indices = tf.where(anchors_iou_max > 0.6, name='rpn_target_positive_indices')  # [:, 0]

    # # 每个GT最大的anchor也是正样本
    # gt_iou_argmax = tf.argmax(iou, axis=1)
    # gt_iou_max = tf.reduce_max(iou, axis=1)

    # 负样本索引号
    negative_indices = tf.where(anchors_iou_max < 0.3, name='rpn_target_negative_indices')  # [:, 0]

    # 正样本
    positive_num = int(rpn_train_anchors * 0.5)  # 正负比例1:1
    positive_num = tf.minimum(positive_num, tf.shape(positive_indices)[0], name='rpn_target_positive_num')
    positive_indices = tf.random_shuffle(positive_indices)[:positive_num]
    positive_anchors = tf.gather_nd(anchors, positive_indices)

    # 负样本
    negative_num = tf.minimum(rpn_train_anchors - positive_num,
                              tf.shape(negative_indices)[0], name='rpn_target_negative_num')
    negative_indices = tf.random_shuffle(negative_indices)[:negative_num]
    # negative_anchors = tf.gather_nd(anchors, negative_indices)

    # 找到正样本对应的GT boxes
    anchors_iou_argmax = tf.argmax(iou, axis=0)  # 每个anchor最大iou对应的GT 索引 [n]
    positive_gt_indices = tf.gather_nd(anchors_iou_argmax, positive_indices)
    # gt_cls，gt_boxes是二维的，使用gather
    positive_gt_boxes = tf.gather(gt_boxes, positive_gt_indices)
    positive_gt_cls = tf.gather(gt_cls, positive_gt_indices)

    # 回归目标
    deltas = regress_target(positive_anchors, positive_gt_boxes)

    # 计算padding
    pad_num = tf.maximum(0, rpn_train_anchors - positive_num - negative_num)
    # 分类正负样本
    negative_gt_cls = tf.stack([tf.ones([negative_num]),
                                tf.zeros([negative_num])], axis=1)  # 负样本稀疏编码[1,0]
    class_ids = tf.concat([positive_gt_cls, negative_gt_cls], axis=0)
    class_ids = tf.pad(class_ids, [[0, pad_num], [0, 0]], name='rpn_target_class_ids')  # padding

    deltas = tf.pad(deltas, [[0, negative_num + pad_num], [0, 0]], name='rpn_target_deltas')

    # 处理索引,记录正负anchor索引位置，第二位为标志位1位前景、0为背景、-1位padding
    positive_part = tf.stack([positive_indices[:, 0], tf.ones([positive_num], dtype=tf.int64)], axis=1)
    negative_part = tf.stack([negative_indices[:, 0], tf.zeros([negative_num], dtype=tf.int64)], axis=1)
    pad_part = tf.ones([pad_num, 2], dtype=tf.int64) * -1
    indices = tf.concat([positive_part, negative_part, pad_part], axis=0, name='rpn_target_indices')

    # 其它统计指标
    gt_num = tf.shape(gt_cls)[0]  # GT数
    miss_match_gt_num = gt_num - tf.shape(tf.unique(positive_gt_indices)[0])[0]  # 未分配anchor的GT

    return class_ids, deltas, indices, tf.cast(gt_num, dtype=tf.float32), tf.cast(positive_num,
                                                                                  dtype=tf.float32), tf.cast(
        miss_match_gt_num, dtype=tf.float32)
    # positive_num, miss_match_gt_num


class RpnTarget(keras.layers.Layer):
    def __init__(self, batch_size, train_anchors_per_image, **kwargs):
        """

        :param batch_size: batch_size大小
        :param train_anchors_per_image: 每张图训练的正负anchors数，不足时零padding填充
        :param kwargs:
        """
        super(RpnTarget, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.train_anchors_per_image = train_anchors_per_image

    def call(self, inputs, **kwargs):
        """
        计算分类和回归目标
        :param inputs:
        inputs[0]: GT 边框坐标 [batch_size, MAX_GT_BOXs,(y1,x1,y2,x2,tag)] ,tag=-1 为padding
        inputs[1]: GT 类别 [batch_size, MAX_GT_BOXs,num_class+1] ;最后一位为tag, tag=-1 为padding
        inputs[2]: Anchors [batch_size, anchor_num,(y1,x1,y2,x2)]
        :param kwargs:
        :return:
        """
        gt_boxes = inputs[0]
        gt_cls_ids = inputs[1]
        anchors = inputs[2]

        outputs = tf_utils.batch_slice(
            [gt_boxes, gt_cls_ids, anchors],
            lambda x, y, z:
            rpn_targets_graph(x, y, z, self.train_anchors_per_image), self.batch_size)

        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.train_anchors_per_image, 2),  # 只有两类
                (input_shape[0][0], self.train_anchors_per_image, 4),
                (input_shape[0][0], self.train_anchors_per_image, 2),
                (input_shape[0][0],),
                (input_shape[0][0],),
                (input_shape[0][0],)]


def apply_regress(deltas, anchors):
    """
    应用回归目标到边框
    :param deltas: 回归目标[N,(dy, dx, dh, dw)]
    :param anchors: anchor boxes[N,(y1,x1,y2,x2)]
    :return:
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    # 中心点坐标
    cy = (anchors[:, 2] + anchors[:, 0]) * 0.5
    cx = (anchors[:, 3] + anchors[:, 1]) * 0.5

    # 回归系数
    deltas *= tf.constant([0.1, 0.1, 0.2, 0.2])
    dy, dx, dh, dw = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 中心坐标回归
    cy += dy * h
    cx += dx * w
    # 高度和宽度回归
    h *= tf.exp(dh)
    w *= tf.exp(dw)

    # 转为y1,x1,y2,x2
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    y2 = cy + h * 0.5
    x2 = cx + w * 0.5

    return tf.stack([y1, x1, y2, x2], axis=1)


def nms(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=0.05, name=None):
    """
    非极大抑制
    :param boxes:
    :param scores:
    :param max_output_size:
    :param iou_threshold:
    :param score_threshold:
    :param name:
    :return: 检测边框、边框得分
    """
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold, score_threshold, name)  # 一维索引
    output_boxes = tf.gather(boxes, indices)  # (M,4)
    class_scores = tf.expand_dims(tf.gather(scores, indices), axis=1)  # 扩展到二维(M,1)
    # padding到固定大小
    return tf_utils.pad_to_fixed_size(output_boxes, max_output_size), \
           tf_utils.pad_to_fixed_size(class_scores, max_output_size)


class RpnToProposal(keras.layers.Layer):
    """
    生成候选框
    """

    def __init__(self, batch_size, score_threshold=0.05, output_box_num=300, iou_threshold=0.5, **kwargs):
        """

        :param batch_size: batch_size
        :param score_threshold: 分数阈值
        :param output_box_num: 生成proposal 边框数量
        :param iou_threshold: nms iou阈值
        """
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.output_box_num = output_box_num
        self.iou_threshold = iou_threshold
        super(RpnToProposal, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        应用边框回归，并使用nms生成最后的边框
        :param inputs:
        inputs[0]: deltas, [N,(dy,dx,dh,dw)]
        inputs[1]: class logits [N,num_classes]
        inputs[2]: anchors [N,(y1,x1,y2,x2)]
        :param kwargs:
        :return:
        """
        deltas = inputs[0]
        class_logits = inputs[1]
        anchors = inputs[2]
        # 转为分类评分
        class_scores = tf.nn.softmax(logits=class_logits, axis=-1)[..., -1]  # 最后一类为前景 (N,)

        # 应用边框回归
        proposals = tf_utils.batch_slice([deltas, anchors], lambda x, y: apply_regress(x, y), self.batch_size)

        # # 非极大抑制
        outputs = tf_utils.batch_slice([proposals, class_scores],
                                       lambda x, y: nms(x, y,
                                                        max_output_size=self.output_box_num,
                                                        iou_threshold=self.iou_threshold,
                                                        score_threshold=self.score_threshold),
                                       self.batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        """
        注意多输出，call返回值必须是列表
        :param input_shape:
        :return:
        """
        return [(input_shape[0][0], self.output_box_num, 4 + 1),
                (input_shape[0][0], self.output_box_num, 1 + 1)]


if __name__ == '__main__':
    sess = tf.Session()
    a = tf.constant([[1, 2, 4, 6], [1, 2, 4, 6], [2, 2, 4, 6]], dtype=tf.float32)
    b = tf.constant([[2, 2, 4, 6], [2, 3, 4, 6]], dtype=tf.float32)
    iou = compute_iou(a, b)
    print(sess.run(iou))
    print(sess.run(tf.nn.softmax(b, axis=-1)))
