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
    :param gt_boxes: GT 边框坐标 [MAX_GT_BOXs, (y1,x1,y2,x2,tag)] ,tag=0 为padding
    :param gt_cls: GT 类别 [MAX_GT_BOXs, 1+1] ;最后一位为tag, tag=0 为padding
    :param anchors: [anchor_num, (y1,x1,y2,x2)]
    :param rpn_train_anchors: 训练样本数(256)
    :return:
    deltas:[rpn_train_anchors,(dy,dx,dh,dw,tag)]：anchor边框回归目标,tag=1 为正样本，tag=0为padding，tag=-1为负样本
    class_ids:[rpn_train_anchors,1+1]: anchor边框分类,tag=1 为正样本，tag=0为padding，tag=-1为负样本
    indices:[rpn_train_anchors,(indices,tag)]: tag=1 为正样本，tag=0为padding，tag=-1为负样本
    """

    # 获取真正的GT,去除标签位
    gt_boxes = tf_utils.remove_pad(gt_boxes)
    gt_cls = tf_utils.remove_pad(gt_cls)[:, 0]  # [N,1]转[N]

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

    # 找到正样本对应的GT boxes
    anchors_iou_argmax = tf.argmax(iou, axis=0)  # 每个anchor最大iou对应的GT 索引 [n]
    positive_gt_indices = tf.gather_nd(anchors_iou_argmax, positive_indices)
    # positive_gt_indices是一维的，使用gather
    positive_gt_boxes = tf.gather(gt_boxes, positive_gt_indices)
    positive_gt_cls = tf.gather(gt_cls, positive_gt_indices)

    # 回归目标计算
    deltas = regress_target(positive_anchors, positive_gt_boxes)

    # 负样本
    negative_num = tf.minimum(rpn_train_anchors - positive_num,
                              tf.shape(negative_indices)[0], name='rpn_target_negative_num')
    negative_indices = tf.random_shuffle(negative_indices)[:negative_num]
    negative_gt_cls = tf.zeros([negative_num])  # 负样本类别id为0
    negative_deltas = tf.zeros([negative_num, 4])

    # 合并正负样本
    deltas = tf.concat([deltas, negative_deltas], axis=0, name='rpn_target_deltas')
    class_ids = tf.concat([positive_gt_cls, negative_gt_cls], axis=0, name='rpn_target_class_ids')
    indices = tf.concat([positive_indices, negative_indices], axis=0, name='rpn_train_anchor_indices')

    # 计算padding
    pad_num = tf.maximum(0, rpn_train_anchors - positive_num - negative_num)
    deltas, class_ids, indices = tf_utils.pad_list_to_fixed_size([deltas, tf.expand_dims(class_ids, 1), indices],
                                                                 pad_num)
    # 将负样本tag标志改为-1;方便后续处理;
    deltas[positive_num:positive_num + negative_num, -1] = -1
    class_ids[positive_num:positive_num + negative_num, -1] = -1
    indices[positive_num:positive_num + negative_num, -1] = -1
    # 其它统计指标
    gt_num = tf.shape(gt_cls)[0]  # GT数
    miss_match_gt_num = gt_num - tf.shape(tf.unique(positive_gt_indices)[0])[0]  # 未分配anchor的GT

    return deltas, class_ids, indices, tf.cast(  # 用作度量的必须是浮点类型
        gt_num, dtype=tf.float32), tf.cast(
        positive_num, dtype=tf.float32), tf.cast(
        miss_match_gt_num, dtype=tf.float32)


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


def shuffle_sample(tensor_list, tensor_size, sample_size):
    """
    对相关的tensor 列表随机协同采样
    :param tensor_list:
    :param tensor_size: tensor尺寸，即第一维大小
    :param sample_size: 采样尺寸
    :return:
    """
    shuffle_indices = tf.range(tensor_size)[:sample_size]
    return [tf.gather(tensor, shuffle_indices) for tensor in tensor_list]


def detect_targets_graph(gt_boxes, gt_class_ids, proposals, train_rois_per_image, roi_positive_ratio):
    """
    每个图像生成检测网络的分类和回归目标
    :param gt_boxes: GT 边框坐标 [MAX_GT_BOXs, (y1,x1,y2,x2,tag)] ,tag=0 为padding
    :param gt_class_ids: GT 类别 [MAX_GT_BOXs, 1+1] ;最后一位为tag, tag=0 为padding
    :param proposals: [N,(y1,x1,y2,x2,tag)] ,tag=0 为padding
    :param train_rois_per_image: 每张图像训练的proposal数量
    :param roi_positive_ratio: proposal正负样本比
    :return:
    """
    # 去除padding
    gt_boxes = tf_utils.remove_pad(gt_boxes)
    gt_class_ids = tf_utils.remove_pad(gt_class_ids)[:, 0]  # 从[N,1]变为[N]
    proposals = tf_utils.remove_pad(proposals)
    # 计算iou
    iou = compute_iou(gt_boxes, proposals)

    # 每个GT边框IoU最大的proposal为正
    gt_iou_argmax = tf.argmax(iou, axis=1)

    # GT和对应的proposal
    gt_boxes_pos_1 = tf.identity(gt_boxes)
    gt_class_ids_pos_1 = tf.identity(gt_class_ids)
    proposal_pos_1 = tf.gather(proposals, gt_iou_argmax)

    # 在接下来的操作之前提出已经被选中的proposal
    indices = tf.unique(gt_iou_argmax)[0]  # 被选中的索引
    all_indices = tf.range(tf.shape[proposals][0])  # 所有的索引
    remainder_indices = tf.setdiff1d(all_indices, indices)[0]  # 剩余的索引
    # 剩余的proposals和iou
    proposals = tf.gather(proposals, remainder_indices)
    iou = tf.gather(iou, remainder_indices, axis=1)

    # 正样本每个proposal 最大的iou,且iou>=0.5
    proposal_iou_max = tf.reduce_max(iou, axis=0)
    proposal_pos_idx = tf.where(proposal_iou_max >= 0.5)  # 正样本proposal对应的索引号,二维

    proposal_iou_argmax = tf.argmax(iou, axis=0)
    gt_pos_idx = tf.gather_nd(proposal_iou_argmax, proposal_pos_idx)  # 对应的GT 索引号，一维的

    gt_boxes_pos_2 = tf.gather(gt_boxes, gt_pos_idx)
    gt_class_ids_pos_2 = tf.gather(gt_class_ids, gt_pos_idx)
    proposal_pos_2 = tf.gather_nd(proposals, proposal_pos_idx)

    # 合并两部分正样本
    gt_boxes_pos = tf.concat([gt_boxes_pos_1, gt_boxes_pos_2], axis=0)
    class_ids = tf.concat([gt_class_ids_pos_1, gt_class_ids_pos_2], axis=0)
    proposal_pos = tf.concat([proposal_pos_1, proposal_pos_2], axis=[0])

    # 计算回归目标
    deltas = regress_target(proposal_pos, gt_boxes_pos)

    # 根据正负样本比确定最终的正样本
    positive_num = tf.minimum(tf.shape[proposal_pos][0], int(train_rois_per_image * roi_positive_ratio))
    deltas, class_ids, proposal_pos = shuffle_sample([deltas, class_ids, proposal_pos],
                                                     tf.shape[proposal_pos][0],
                                                     positive_num)

    # 负样本：与所有GT的iou<0.5
    proposal_neg_idx = tf.where(proposal_iou_max < 0.5)
    proposal_neg = tf.gather_nd(proposals, proposal_neg_idx)

    negative_num = tf.minimum(train_rois_per_image - positive_num, tf.shape(proposal_neg)[0])

    class_ids_neg = tf.zeros(shape=[negative_num])  # 背景类，类别id为0
    deltas_neg = tf.zeros(shape=[negative_num, 4])

    # 合并正负样本
    train_rois = tf.concat([proposal_pos, proposal_neg], axis=0)
    deltas = tf.concat([deltas, deltas_neg], axis=0)
    class_ids = tf.concat([class_ids, class_ids_neg], axis=0)

    # 计算padding
    pad_num = train_rois_per_image - positive_num - negative_num
    deltas, class_ids, train_rois = tf_utils.pad_list_to_fixed_size(
        [deltas, tf.expand_dims(class_ids, axis=1), train_rois], pad_num)  # class_ids分类扩一维
    # 为后续处理方便负样本tag设置为-1
    deltas[positive_num:positive_num + negative_num, -1] = -1

    return deltas, class_ids, train_rois


class DetectTarget(keras.layers.Layer):
    """
    检测网络分类和回归目标;同时还要过滤训练的proposals
    """

    def __init__(self, batch_size, train_rois_per_image=200, roi_positive_ratio=0.33, **kwargs):
        self.batch_size = batch_size
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        super(DetectTarget, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        计算检测分类和回归目标
        :param inputs:
        inputs[0]: GT 边框坐标 [batch_size, MAX_GT_BOXs,(y1,x1,y2,x2,tag)] ,tag=0 为padding
        inputs[1]: GT 类别 [batch_size, MAX_GT_BOXs,num_class+1] ;最后一位为tag, tag=0 为padding
        inputs[2]: proposals [batch_size, N,(y1,x1,y2,x2)]
        :param kwargs:
        :return: [deltas,class_ids,rois]
        """
        gt_boxes = inputs[0]
        gt_class_ids = inputs[1]
        proposals = inputs[2]

        outputs = tf_utils.batch_slice([gt_boxes, gt_class_ids, proposals],
                                       lambda x, y, z: detect_targets_graph(x, y, z, self.train_rois_per_image,
                                                                            self.roi_positive_ratio),
                                       self.batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.train_rois_per_image, 4 + 1),  # deltas
                (input_shape[0][0], self.train_rois_per_image, 1 + 1),  # class_ids
                (input_shape[0][0], self.train_rois_per_image, 4 + 1)]  # rois


def main():
    sess = tf.Session()
    # a = tf.constant([[1, 2, 4, 6], [1, 2, 4, 6], [2, 2, 4, 6]], dtype=tf.float32)
    # b = tf.constant([[2, 2, 4, 6], [2, 3, 4, 6]], dtype=tf.float32)
    # iou = compute_iou(a, b)
    # print(sess.run(iou))
    # print(sess.run(tf.nn.softmax(b, axis=-1)))

    x = tf.range(10)
    y = tf.constant([3, 2, 1])
    diff = tf.setdiff1d(x, y)
    print(sess.run(diff[0]))
    print(sess.run(tf.unique(y)))


if __name__ == '__main__':
    main()
