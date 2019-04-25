# -*- coding: utf-8 -*-
"""
Created on 2019/3/24 下午9:51
基础网络

@author: mick.yi

"""
from keras import layers, backend, Model
from keras.layers import TimeDistributed, Conv2D


def resnet50(inputs):
    bn_axis = 3
    #
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # # 确定精调层
    no_train_model = Model(inputs=input, outputs=x)
    for l in no_train_model.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = True
        else:
            l.trainable = False

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # model = Model(input, x, name='resnet50')

    return x


def resnet50_head(features):
    filters = 512
    x = features
    x = conv_block_5d(x, 3, [filters, filters, filters * 4], stage=5, block='a', strides=(2, 2))
    x = identity_block_5d(x, 3, [filters, filters, filters * 4], stage=5, block='b')
    x = identity_block_5d(x, 3, [filters, filters, filters * 4], stage=5, block='c')
    # 全局平均池化(batch_size,roi_num,channels)
    x = layers.TimeDistributed(layers.GlobalAvgPool2D())(x)
    return x


def vgg16(inputs):
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    return x


def vgg16_head(features, fc_layers_size=4096):
    """

    :param features: [batch_size,rois_num,H,W,C]
    :param fc_layers_size:
    :return:
    """
    # 打平
    x = TimeDistributed(layers.Flatten(features))  # [batch_size,rois_num,H*W*C]
    # fc6
    x = TimeDistributed(layers.Dense(fc_layers_size), name='fc1')(x)  # 变为(batch_size,roi_num,channels)
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(rate=0.5, name='drop_fc6')(x)

    x = TimeDistributed(layers.Dense(fc_layers_size), name='f2')(x)  # 变为(batch_size,roi_num,channels)
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(rate=0.5, name='drop_fc7')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def identity_block_5d(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.TimeDistributed(layers.Conv2D(filters1, (1, 1),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2a')(input_tensor)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=-1), name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(filters2, kernel_size,
                                             padding='same',
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2b')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(filters3, (1, 1),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2c')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block_5d(input_tensor,
                  kernel_size,
                  filters,
                  stage,
                  block,
                  strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.TimeDistributed(layers.Conv2D(filters1, (1, 1), strides=strides,
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2a')(input_tensor)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(filters2, kernel_size, padding='same',
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2b')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(filters3, (1, 1),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2c')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = layers.TimeDistributed(layers.Conv2D(filters3, (1, 1), strides=strides,
                                                    kernel_initializer='he_normal'),
                                      name=conv_name_base + '1')(input_tensor)
    shortcut = layers.TimeDistributed(layers.BatchNormalization(
        axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x
