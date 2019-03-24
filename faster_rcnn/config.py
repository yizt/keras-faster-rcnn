"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes
    # 网络步长
    BACKBONE_STRIDE = 16
    # anchors
    RPN_ANCHOR_BASE_SIZE = 64
    RPN_ANCHOR_SCALES = [1, 2 ** 1, 2 ** 2]
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_NUM = len(RPN_ANCHOR_SCALES) * len(RPN_ANCHOR_RATIOS)

    # RPN提议框非极大抑制阈值(训练时可以增加该值来增加提议框)
    RPN_NMS_THRESHOLD = 0.7

    # 每张图像训练anchors个数
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # 训练参数
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # 权重衰减
    WEIGHT_DECAY = 0.0001
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
    # Give the configuration a recognizable name
    NAME = "voc"

    IMAGE_MIN_DIM = 608
    IMAGE_MAX_DIM = 608
    IMAGE_INPUT_SHAPE = (IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3)

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    BATCH_SIZE = IMAGES_PER_GPU

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # voc has 20 classes
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


class MacVoConfig():
    voc_path = '/Users/yizuotian/dataset/VOCdevkit/'
    pretrained_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# 当前配置
current_config = VOCConfig()
