import numpy as np


class Config(object):
    NAME = None
    # ###输入参数####
    # 类别数
    NUM_CLASSES = 1
    # 输入图像大小
    IMAGE_MAX_DIM = 608
    IMAGE_INPUT_SHAPE = (IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3)
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
    RPN_ANCHOR_NUM = len(RPN_ANCHOR_HEIGHTS) if RPN_ANCHOR_HEIGHTS is not None \
        else len(RPN_ANCHOR_SCALES) * len(RPN_ANCHOR_RATIOS)

    # RPN提议框非极大抑制阈值(训练时可以增加该值来增加提议框)
    RPN_NMS_THRESHOLD = 0.7
    # 每张图像训练anchors个数
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # 训练和预测阶段NMS后保留的ROIs数
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # 检测网络训练rois数和正样本比
    TRAIN_ROIS_PER_IMAGE = 200
    ROI_POSITIVE_RATIO = 0.33

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
    IMAGES_PER_GPU = 2
    BATCH_SIZE = IMAGES_PER_GPU
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
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


class VOCConfig(Config):
    NAME = "voc"
    NUM_CLASSES = 1 + 20  #
    IMAGE_MAX_DIM = 720
    LEARNING_RATE = 0.01
    IMAGE_INPUT_SHAPE = (720, 720, 3)
    RPN_ANCHOR_HEIGHTS = [258.15, 87.23, 226.65, 386.09, 491.17]
    RPN_ANCHOR_WIDTHS = [163.03, 73.11, 369.82, 617.81, 331.97]
    RPN_ANCHOR_NUM = len(RPN_ANCHOR_HEIGHTS)
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

    pretrained_weights = '/opt/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    rpn_weights = '/tmp/frcnn-rpn.h5'
    rcnn_weights = '/tmp/frcnn-rcnn.h5'
    voc_path = '/opt/dataset/VOCdevkit'


class LocalVOCConfig(VOCConfig):
    pretrained_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights = 'frcnn-rpn.010.h5'
    voc_path = r'd:\work\图像识别\VOCtrainval_06-Nov-2007\VOCdevkit'


class MacVoConfig(VOCConfig):
    voc_path = '/Users/yizuotian/dataset/VOCdevkit/'
    pretrained_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# 当前配置
current_config = VOCConfig()
