#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-02-17 11:03:35
#   Description :
#
#================================================================

import core.common as common
import tensorflow as tf
import numpy as np
def darknet53(input_data, trainable=True):

    with tf.variable_scope('darknet53'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data


def darknet19(input_data,trainable=True):
    with tf.variable_scope('darknet19'):

        input_data = common.convolutional(input_data,filters_shape =(3, 3, 3, 16),trainable=trainable,name='conv1')
       
        input_data = common.maxpool(input_data,name='pool1') # /2

        input_data = common.convolutional(input_data,filters_shape =(3, 3, 16, 32),trainable=trainable,name='conv2')
     
        input_data = common.maxpool(input_data,name='pool2') # /4

        input_data = common.convolutional(input_data,filters_shape =(3, 3, 32, 64),trainable=trainable,name='conv3_1')
        input_data = common.convolutional(input_data,filters_shape =(1, 1, 64, 32),trainable=trainable,name='conv3_2')
        input_data = common.convolutional(input_data,filters_shape =(3, 3, 32, 64),trainable=trainable,name='conv3_3')
        input_data = common.maxpool(input_data,name='pool3') # /8
        l_feature  = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 128),trainable=trainable, name='conv4_1')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 128, 64), trainable=trainable,name='conv4_2')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 128),trainable=trainable, name='conv4_3')
        input_data = common.maxpool(input_data,name='pool4') # /16
        m_feature  = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),trainable=trainable, name='conv5_1')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 256, 128),trainable=trainable, name='conv5_2')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),trainable=trainable, name='conv5_3')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 256, 128),trainable=trainable, name='conv5_4')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),trainable=trainable, name='conv5_5')
        input_data = common.maxpool(input_data,name='pool5') # /32

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512), trainable=trainable,name='conv6_1')
        input_data = common.convolutional(input_data, filters_shape=(1 ,1, 512, 256),trainable=trainable, name='conv6_2')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),trainable=trainable, name='conv6_3')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 512, 256),trainable=trainable, name='conv6_4')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512), trainable=trainable,name='conv6_5')
        s_feature  = input_data
        return l_feature,m_feature,s_feature

