#/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 09:56:29
#   Description :
#
#================================================================

import tensorflow as tf
import tensorflow.contrib.slim as slim

def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)
        if bn:
           conv=slim.batch_norm(conv,scale=True,is_training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.relu(conv)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]
    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    elif  method == "deconv":
         numm_filter = input_data.shape.as_list()[-1]
    
         h=input_data.shape.as_list()[1]
   
         w=input_data.shape.as_list()[2]
         batch=input_data.shape.as_list()[0]
         weight= tf.get_variable('conv_weight'+name, shape=[4, 4, numm_filter, numm_filter])
         
         output = tf.nn.conv2d_transpose(input_data, weight,[batch,h*2,w*2,numm_filter], strides=[1,2,2,1], padding='SAME',name=name)
    return output


def maxpool(input_data,k=2,s=2,padding='SAME',name=None):
    return tf.nn.max_pool(input_data,(1,k,k,1),(1,s,s,1),padding,name='max_pool')

def depth_wise_conv(input,kernel_size,stride=1,name=None):
    input_shape=input.shape.as_list()
    with tf.variable_scope(name):
        w=tf.get_variable("w",shape=(kernel_size,kernel_size,input_shape[-1],1),dtype=tf.float32)
        out=tf.nn.depthwise_conv2d(input,w,strides=[1,stride,stride,1],padding="SAME")
    return out
