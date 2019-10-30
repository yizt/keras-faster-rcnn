# -*- coding: utf-8 -*-
"""
Created on 2018/12/2 下午3:43

@author: mick.yi

分类回归目标层，包括rpn_target和detect_target

"""
import tensorflow.python.keras as keras
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


def rpn_targets_graph(gt_boxes, gt_cls, anchors, anchors_tag, rpn_train_anchors=None):
    """
    处理单个图像的rpn分类和回归目标
    a)正样本为 IoU>0.7的anchor;负样本为IoU<0.3的anchor; 居中的为中性样本，丢弃
    b)需要保证所有的GT都有anchor对应，即使IoU<0.3；
    c)正负样本比例保持1:1
    :param gt_boxes: GT 边框坐标 [MAX_GT_BOXs, (y1,x1,y2,x2,tag)] ,tag=0 为padding
    :param gt_cls: GT 类别 [MAX_GT_BOXs, 1+1] ;最后一位为tag, tag=0 为padding
    :param anchors: [anchor_num, (y1,x1,y2,x2)]
    :param anchors_tag:[anchor_num] bool类型
    :param rpn_train_anchors: 训练样本数(256)
    :return:
    deltas:[rpn_train_anchors,(dy,dx,dh,dw,tag)]：anchor边框回归目标,tag=1 为正样本，tag=0为padding，tag=-1为负样本
    class_ids:[rpn_train_anchors,1+1]: anchor边框分类,tag=1 为正样本，tag=0为padding，tag=-1为负样本
    indices:[rpn_train_anchors,(indices,tag)]: tag=1 为正样本，tag=0为padding，tag=-1为负样本
    """

    # 获取真正的GT,去除标签位
    gt_boxes = tf_utils.remove_pad(gt_boxes)
    gt_cls = tf_utils.remove_pad(gt_cls)[:, 0]  # [N,1]转[N]

    # 获取有效的anchors
    valid_anchor_indices = tf.where(anchors_tag)[:, 0]  # [valid_anchors_num]
    anchors = tf.gather(anchors, valid_anchor_indices)
    # 计算IoU
    iou = compute_iou(gt_boxes, anchors)
    # print("iou:{}".format(iou))

    # 每个GT对应的IoU最大的anchor是正样本
    gt_iou_argmax = tf.argmax(iou, axis=1)
    positive_gt_indices_1 = tf.range(tf.shape(gt_boxes)[0])  # 索引号就是1..n-1
    positive_anchor_indices_1 = gt_iou_argmax

    # 每个anchors最大iou ，且iou>0.7的为正样本
    anchors_iou_max = tf.reduce_max(iou, axis=0)
    # 正样本索引号（iou>0.7),
    positive_anchor_indices_2 = tf.where(anchors_iou_max > 0.7, name='rpn_target_positive_indices')  # [:, 0]
    # 找到正样本对应的GT boxes 索引
    # anchors_iou_argmax = tf.argmax(iou, axis=0)  # 每个anchor最大iou对应的GT 索引 [n]
    anchors_iou_argmax = tf.cond(  # 需要考虑GT个数为0的情况
        tf.greater(tf.shape(gt_boxes)[0], 0),
        true_fn=lambda: tf.argmax(iou, axis=0),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    positive_gt_indices_2 = tf.gather_nd(anchors_iou_argmax, positive_anchor_indices_2)

    # 合并两部分正样本
    positive_gt_indices = tf.concat([positive_gt_indices_1, tf.cast(positive_gt_indices_2, tf.int32)], axis=0,
                                    name='rpn_gt_boxes_concat')
    positive_anchor_indices = tf.concat([positive_anchor_indices_1, positive_anchor_indices_2[:, 0]], axis=0,
                                        name='rpn_positive_anchors_concat')

    # 根据正负样本比1:1,确定最终的正样本
    positive_num = tf.minimum(tf.shape(positive_anchor_indices)[0], int(rpn_train_anchors * 0.9))
    positive_anchor_indices, positive_gt_indices = shuffle_sample(
        [positive_anchor_indices, positive_gt_indices],
        tf.shape(positive_anchor_indices)[0],
        positive_num)
    # 根据索引选择anchor和GT
    positive_anchors = tf.gather(anchors, positive_anchor_indices)
    positive_gt_boxes = tf.gather(gt_boxes, positive_gt_indices)
    positive_gt_cls = tf.gather(gt_cls, positive_gt_indices)
    # 回归目标计算
    deltas = regress_target(positive_anchors, positive_gt_boxes)

    # 处理负样本
    negative_indices = tf.where(anchors_iou_max < 0.3, name='rpn_target_negative_indices')  # [:, 0]

    # 负样本,保证负样本不超过一半
    negative_num = tf.minimum(rpn_train_anchors - positive_num,
                              tf.shape(negative_indices)[0], name='rpn_target_negative_num')
    # negative_num = tf.minimum(int(rpn_train_anchors * 0.5), negative_num, name='rpn_target_negative_num_2')
    negative_indices = tf.random_shuffle(negative_indices)[:negative_num]
    negative_gt_cls = tf.zeros([negative_num])  # 负样本类别id为0
    negative_deltas = tf.zeros([negative_num, 4])

    # 合并正负样本
    deltas = tf.concat([deltas, negative_deltas], axis=0, name='rpn_target_deltas')
    class_ids = tf.concat([positive_gt_cls, negative_gt_cls], axis=0, name='rpn_target_class_ids')
    indices = tf.concat([positive_anchor_indices, negative_indices[:, 0]], axis=0, name='rpn_train_anchor_indices')

    # indices转换会原始的anchors索引
    indices = tf.gather(valid_anchor_indices, indices, name='map_to_origin_anchor_indices')
    # 计算padding
    deltas, class_ids = tf_utils.pad_list_to_fixed_size([deltas, tf.expand_dims(class_ids, 1)],
                                                        rpn_train_anchors)
    # 将负样本tag标志改为-1;方便后续处理;
    indices = tf_utils.pad_to_fixed_size_with_negative(tf.expand_dims(indices, 1), rpn_train_anchors,
                                                       negative_num=negative_num, data_type=tf.int64)
    # 其它统计指标
    gt_num = tf.shape(gt_cls)[0]  # GT数
    miss_match_gt_num = gt_num - tf.shape(tf.unique(positive_gt_indices)[0])[0]  # 未分配anchor的GT
    rpn_gt_min_max_iou = tf.reduce_min(tf.reduce_max(iou, axis=1))  # GT匹配anchor最小的IoU

    return [deltas, class_ids, indices,
            tf_utils.scalar_to_1d_tensor(gt_num),
            tf_utils.scalar_to_1d_tensor(positive_num),
            tf_utils.scalar_to_1d_tensor(negative_num),
            tf_utils.scalar_to_1d_tensor(miss_match_gt_num),
            tf_utils.scalar_to_1d_tensor(rpn_gt_min_max_iou)]


class RpnTarget(keras.layers.Layer):
    def __init__(self, batch_size, train_anchors_per_image, **kwargs):
        """

        :param batch_size: batch_size大小
        :param train_anchors_per_image: 每张图训练的正负anchors数，不足时零padding填充
        :param kwargs:
        """
        super(RpnTarget, self).__init__(**kwargs)
        self.train_anchors_per_image = train_anchors_per_image
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        """
        计算分类和回归目标
        :param inputs:
        inputs[0]: GT 边框坐标 [batch_size, MAX_GT_BOXs,(y1,x1,y2,x2,tag)] ,tag=-1 为padding
        inputs[1]: GT 类别 [batch_size, MAX_GT_BOXs,num_class+1] ;最后一位为tag, tag=-1 为padding
        inputs[2]: Anchors [batch_size, anchor_num,(y1,x1,y2,x2)]
        inputs[3]: Anchors是否有效 bool类型 [batch_size, anchor_num]
        :param kwargs:
        :return:
        """
        gt_boxes = inputs[0]
        gt_cls_ids = inputs[1]
        anchors = inputs[2]
        anchors_tag = inputs[3]

        # options = {"rpn_train_anchors": self.train_anchors_per_image}
        # outputs = tf.map_fn(lambda x: rpn_targets_graph(*x, **options),
        #                    elems=[gt_boxes, gt_cls_ids, anchors],
        #                    dtype=[tf.float32] * 2 + [tf.int64] + [tf.float32] * 4)

        outputs = tf_utils.batch_slice(
            [gt_boxes, gt_cls_ids, anchors, anchors_tag],
            lambda x, y, z, t:
            rpn_targets_graph(x, y, z, t, self.train_anchors_per_image), self.batch_size)

        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.train_anchors_per_image, 5),
                (input_shape[0][0], self.train_anchors_per_image, 2),  # 只有两类
                (input_shape[0][0], self.train_anchors_per_image, 2),
                (input_shape[0][0], 1),
                (input_shape[0][0], 1),
                (input_shape[0][0], 1),
                (input_shape[0][0], 1),
                (input_shape[0][0], 1)]


def shuffle_sample(tensor_list, tensor_size, sample_size):
    """
    对相关的tensor 列表随机协同采样
    :param tensor_list:
    :param tensor_size: tensor尺寸，即第一维大小
    :param sample_size: 采样尺寸
    :return:
    """
    shuffle_indices = tf.random_shuffle(tf.range(tensor_size))[:sample_size]
    return [tf.gather(tensor, shuffle_indices) for tensor in tensor_list]


def detect_targets_graph(gt_boxes, gt_class_ids, proposals, train_rois_per_image, roi_positive_ratio):
    """
    每个图像生成检测网络的分类和回归目标
    IoU>=0.5的为正样本；IoU<0.5的为负样本
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
    proposals_num = tf.shape(proposals)[0]
    # 计算iou
    iou = compute_iou(gt_boxes, proposals)  # [gt_num,rois_num]

    # iou >=0.5为正
    proposals_iou_max = tf.reduce_max(iou, axis=0)  # [rois_num]
    positive_indices = tf.where(tf.logical_and(tf.equal(iou, proposals_iou_max),
                                               tf.greater_equal(iou, 0.5)))
    gt_pos_idx = positive_indices[:, 0]  # 第一维gt索引
    proposal_pos_idx = positive_indices[:, 1]  # 第二位rois索引
    match_gt_num = tf.shape(tf.unique(gt_pos_idx)[0])[0]  # shuffle 前匹配的gt num

    gt_boxes_pos = tf.gather(gt_boxes, gt_pos_idx)
    class_ids = tf.gather(gt_class_ids, gt_pos_idx)
    proposal_pos = tf.gather(proposals, proposal_pos_idx)
    # 根据正负样本比确定最终的正样本
    positive_num = tf.minimum(tf.shape(proposal_pos)[0], int(train_rois_per_image * roi_positive_ratio))
    gt_boxes_pos, class_ids, proposal_pos, gt_pos_idx = shuffle_sample(
        [gt_boxes_pos, class_ids, proposal_pos, gt_pos_idx],
        tf.shape(proposal_pos)[0],
        positive_num)
    match_gt_num_after_shuffle = tf.shape(tf.unique(gt_pos_idx)[0])[0]  # shuffle 后匹配的gt num

    # 计算回归目标
    deltas = regress_target(proposal_pos, gt_boxes_pos)

    # 负样本：与所有GT的iou<0.5且iou>0.1
    proposal_iou_max = tf.reduce_max(iou, axis=0)
    proposal_neg_idx = tf.cond(  # 需要考虑GT个数为0的情况;全部都是负样本
        tf.greater(tf.shape(gt_boxes)[0], 0),
        true_fn=lambda: tf.where(tf.logical_and(proposal_iou_max < 0.5, proposal_iou_max > 0.1))[:, 0],
        false_fn=lambda: tf.cast(tf.range(proposals_num), dtype=tf.int64)
    )
    # 确定负样本数量
    negative_num = tf.minimum(train_rois_per_image - positive_num, tf.shape(proposal_neg_idx)[0])
    proposal_neg_idx = tf.random_shuffle(proposal_neg_idx)[:negative_num]
    # 收集负样本
    proposal_neg = tf.gather(proposals, proposal_neg_idx)
    class_ids_neg = tf.zeros(shape=[negative_num])  # 背景类，类别id为0
    deltas_neg = tf.zeros(shape=[negative_num, 4])

    # 合并正负样本
    train_rois = tf.concat([proposal_pos, proposal_neg], axis=0)
    deltas = tf.concat([deltas, deltas_neg], axis=0)
    class_ids = tf.concat([class_ids, class_ids_neg], axis=0)

    # 计算padding
    class_ids, train_rois = tf_utils.pad_list_to_fixed_size(
        [tf.expand_dims(class_ids, axis=1), train_rois], train_rois_per_image)  # class_ids分类扩一维
    # 为后续处理方便负样本tag设置为-1
    deltas = tf_utils.pad_to_fixed_size_with_negative(deltas, train_rois_per_image, negative_num=negative_num)
    # 其它统计指标
    gt_num = tf.shape(gt_class_ids)[0]  # GT数
    miss_gt_num = gt_num - match_gt_num
    miss_gt_num_shuffle = gt_num - match_gt_num_after_shuffle  # shuffle后未分配roi的GT
    gt_min_max_iou = tf.reduce_min(tf.reduce_max(iou, axis=1))  # gt 匹配最小最大值
    return [deltas, class_ids, train_rois,
            tf_utils.scalar_to_1d_tensor(miss_gt_num),
            tf_utils.scalar_to_1d_tensor(miss_gt_num_shuffle),
            tf_utils.scalar_to_1d_tensor(gt_min_max_iou),
            tf_utils.scalar_to_1d_tensor(positive_num),
            tf_utils.scalar_to_1d_tensor(negative_num),
            tf_utils.scalar_to_1d_tensor(proposals_num)]


class DetectTarget(keras.layers.Layer):
    """
    检测网络分类和回归目标;同时还要过滤训练的proposals
    """

    def __init__(self, batch_size, train_rois_per_image=200, roi_positive_ratio=0.33, **kwargs):
        self.batch_size = batch_size
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        super(DetectTarget, self).__init__(**kwargs)
        # print("roi_positive_ratio：{}".format(roi_positive_ratio))

    def call(self, inputs, **kwargs):
        """
        计算检测分类和回归目标
        :param inputs:
        inputs[0]: GT 边框坐标 [batch_size, MAX_GT_BOXs,(y1,x1,y2,x2,tag)] ,tag=0 为padding
        inputs[1]: GT 边框 类别 [batch_size, MAX_GT_BOXs,1+1] ;最后一位为tag, tag=0 为padding
        inputs[2]: proposals [batch_size, N,(y1,x1,y2,x2,tag)]
        :param kwargs:
        :return: [deltas,class_ids,rois]
        """
        gt_boxes = inputs[0]
        gt_class_ids = inputs[1]
        proposals = inputs[2]

        # options = {"train_rois_per_image": self.train_rois_per_image,
        #           "roi_positive_ratio": self.roi_positive_ratio}
        # outputs = tf.map_fn(lambda x: detect_targets_graph(*x, **options),
        #                    elems=[gt_boxes, gt_class_ids, proposals],
        #                    dtype=[tf.float32] * 4)
        outputs = tf_utils.batch_slice([gt_boxes, gt_class_ids, proposals],
                                       lambda x, y, z: detect_targets_graph(x, y, z, self.train_rois_per_image,
                                                                            self.roi_positive_ratio),
                                       self.batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.train_rois_per_image, 4 + 1),  # deltas
                (input_shape[0][0], self.train_rois_per_image, 1 + 1),  # class_ids
                (input_shape[0][0], self.train_rois_per_image, 4 + 1),
                (input_shape[0][0], 1),  # miss_match_gt_num
                (input_shape[0][0], 1),  # miss_match_gt_num after shuffle
                (input_shape[0][0], 1),  # gt_min_max_iou
                (input_shape[0][0], 1),  # positive_roi_num
                (input_shape[0][0], 1),  # negative_roi_num
                (input_shape[0][0], 1)]  # roi_num


def main():
    sess = tf.Session()
    # a = tf.constant([[1, 2, 4, 6], [1, 2, 4, 6], [2, 2, 4, 6]], dtype=tf.float32)
    # b = tf.constant([[2, 2, 4, 6], [2, 3, 4, 6]], dtype=tf.float32)
    # iou = compute_iou(a, b)
    # print(sess.run(iou))
    # print(sess.run(tf.nn.softmax(b, axis=-1)))

    # x = tf.range(10)
    # y = tf.constant([3, 2, 1])
    # diff = tf.setdiff1d(x, y)
    # print(sess.run(diff[0]))
    # print(sess.run(tf.unique(y)))
    x = tf.Variable(tf.ones((3, 4, 5)))
    y = x[1]
    sess.run(tf.global_variables_initializer())

    with tf.control_dependencies([tf.assign(y, tf.zeros((4, 5)))]):
        sess.run(tf.identity(y))
        print(sess.run(x))
        print(sess.run(y))
        tf.scatter_update


if __name__ == '__main__':
    main()
