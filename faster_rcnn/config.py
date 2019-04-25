import numpy as np
from faster_rcnn.layers.base_net import vgg16, vgg16_head, resnet50, resnet50_head


class Config(object):
    NAME = None
    # ###输入参数####
    # 类别数
    NUM_CLASSES = 1
    # 输入图像大小
    IMAGE_MAX_DIM = 608
    # 最大GT个数
    MAX_GT_INSTANCES = 50
    # ####网络参数######
    # 网络步长
    BACKBONE_STRIDE = 16
    #  ROIs 池化大小
    POOL_SIZE = (7, 7)
    # RCNN网络全连接层大小
    RCNN_FC_LAYERS_SIZE = 1024
    # #####anchors######
    RPN_ANCHOR_HEIGHTS = None
    RPN_ANCHOR_WIDTHS = None
    # 指定了anchor长宽，就不用指定base_size,scale,ratio
    RPN_ANCHOR_BASE_SIZE = 64
    RPN_ANCHOR_SCALES = [1, 2 ** 1, 2 ** 2]
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # RPN提议框非极大抑制阈值(训练时可以增加该值来增加提议框)
    RPN_NMS_THRESHOLD_TRAINING = 0.7
    RPN_NMS_THRESHOLD_INFERENCE = 0.7
    # 每张图像训练anchors个数
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # 训练和预测阶段NMS后保留的ROIs数
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # 检测网络训练rois数和正样本比
    TRAIN_ROIS_PER_IMAGE = 128
    ROI_POSITIVE_RATIO = 0.25

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # RPN和RCNN网络边框回归标准差
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # ######输出参数##########
    # 输出检测边框个数
    DETECTION_MAX_INSTANCES = 100
    # 输出检测边框最小置信度
    DETECTION_MIN_CONFIDENCE = 0.7
    # 类别相关检测边框NMS阈值
    DETECTION_NMS_THRESHOLD = 0.3

    # ####训练参数#######
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    TRAIN_LAYERS = r".*"
    # 权重衰减
    WEIGHT_DECAY = 0.0005
    # 梯度裁剪
    GRADIENT_CLIP_NORM = 1.0

    # 损失函数权重
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "rcnn_class_loss": 1.,
        "rcnn_bbox_loss": 1.
    }
    # 数据增广
    USE_HORIZONTAL_FLIP = False
    USE_RANDOM_CROP = False

    def __init__(self):
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT  # batch_size是GPU数乘每个gpu处理图片数
        self.IMAGE_INPUT_SHAPE = (self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3)
        # 如果知道长宽就用指定的，没有指定就使用尺寸个数乘缩放比个数
        self.RPN_ANCHOR_NUM = len(self.RPN_ANCHOR_HEIGHTS) if self.RPN_ANCHOR_HEIGHTS is not None \
            else len(self.RPN_ANCHOR_SCALES) * len(self.RPN_ANCHOR_RATIOS)

    BASE_NET_NAME = 'vgg16'

    def base_fn(self, *args, **kwargs):
        return vgg16(*args, **kwargs)

    def head_fn(self, *args, **kwargs):
        return vgg16_head(*args, **kwargs)


class VOCConfig(Config):
    NUM_CLASSES = 1 + 20  #
    CLASS_MAPPING = {'bg': 0,
                     'train': 1,
                     'dog': 2,
                     'bicycle': 3,
                     'bus': 4,
                     'car': 5,
                     'person': 6,
                     'bird': 7,
                     'chair': 8,
                     'diningtable': 9,
                     'sheep': 10,
                     'tvmonitor': 11,
                     'horse': 12,
                     'sofa': 13,
                     'bottle': 14,
                     'cat': 15,
                     'cow': 16,
                     'pottedplant': 17,
                     'boat': 18,
                     'motorbike': 19,
                     'aeroplane': 20
                     }
    voc_path = '/opt/dataset/VOCdevkit'
    pretrained_weights = None


class VOCVggConfig(VOCConfig):
    NAME = "voc"
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_BASE_SIZE = 128
    LEARNING_RATE = 0.003
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 80
    RPN_NMS_THRESHOLD_TRAINING = 0.8  # 增加训练时rcnn网络正样本数量
    POST_NMS_ROIS_INFERENCE = 300  # 加快预测速度

    USE_HORIZONTAL_FLIP = True
    TRAIN_LAYERS = r"(block3.*)|(block4.*)|(block5.*)|(fc.*)|(rcnn\_.*)|(rpn\_.*)"
    # USE_RANDOM_CROP = True
    pretrained_weights = '/opt/pretrained_model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


class VOCResnetConfig(VOCConfig):
    NAME = "voc"
    BASE_NET_NAME = 'resnet50'
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2
    IMAGE_MAX_DIM = 720
    LEARNING_RATE = 0.003
    TRAIN_LAYERS = r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(rcnn\_.*)|(rpn\_.*)"
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 80
    RPN_NMS_THRESHOLD_TRAINING = 0.8  # 增加训练时rcnn网络正样本数量
    POST_NMS_ROIS_INFERENCE = 300  # 加快预测速度
    POOL_SIZE = (14, 14)

    # RPN_ANCHOR_HEIGHTS = [76.01, 137.64, 210.27, 249.25, 350.94, 386.97, 546.33, 631.71, 707.87]
    # RPN_ANCHOR_WIDTHS = [59.84, 192.22, 98.73, 358.32, 696.61, 195.61, 348.98, 891.01, 566.18]
    RPN_ANCHOR_HEIGHTS = [52.42, 85.64, 143.89, 186.92, 208.26, 266.1, 359.72, 446.26, 484.92]
    RPN_ANCHOR_WIDTHS = [40.85, 132.84, 66.24, 294.26, 135.53, 533.3, 190.26, 339.55, 591.88]

    USE_HORIZONTAL_FLIP = True
    # USE_RANDOM_CROP = True

    BASE_NET_NAME = 'resnet50'

    def base_fn(self, *args, **kwargs):
        return resnet50(*args, **kwargs)

    def head_fn(self, *args, **kwargs):
        return resnet50_head(*args, **kwargs)

    pretrained_weights = '/opt/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


class LocalVOCConfig(VOCConfig):
    GPU_COUNT = 1
    pretrained_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights = 'frcnn-rpn.010.h5'
    voc_path = r'd:\work\图像识别\VOCtrainval_06-Nov-2007\VOCdevkit'


class MacVoConfig(VOCConfig):
    GPU_COUNT = 1
    voc_path = '/Users/yizuotian/dataset/VOCdevkit/'
    pretrained_weights = '/Users/yizuotian/dataset/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# 当前配置
current_config = VOCResnetConfig()

if __name__ == '__main__':
    print("batch_size:{}".format(current_config.BATCH_SIZE))
    print("input_shape:{}".format(current_config.IMAGE_INPUT_SHAPE))
    print("train layers:{}".format(current_config.TRAIN_LAYERS))
