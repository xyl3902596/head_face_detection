#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 10:47:03
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, trainable,train_backbone):

        self.trainable        = trainable
        # self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.classes=['front','rear']
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.YOLO.STRIDES)
        self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD

        self.conv_sbbox, self.conv_mbbox, self.conv_lbbox=self.__build_nework(input_data,train_backbone)


        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[2], self.strides[2])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[0], self.strides[0])

    def __build_nework(self, input_data,train_backbone):
        if train_backbone==0:
            route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)
            input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
            input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
            input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
            input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
            input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

            conv_sobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable,'conv_sobj_branch',)
            conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 1024, 3*(self.num_class + 9)),
                                              trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
            input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

            with tf.variable_scope('route_1'):
                input_data = tf.concat([input_data, route_2], axis=3)

            input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
            input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
            input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
            input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
            input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

            conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable,'conv_mobj_branch',)
            conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 9)),
                                              trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
            input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

            with tf.variable_scope('route_2'):
                input_data = tf.concat([input_data, route_1], axis=3)

            input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
            input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
            input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

            conv_lobj_branch = common.convolutional(input_data, (3, 3, 128, 256),self.trainable,'conv_lobj_branch')
            conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 256, 3*(self.num_class + 9)),
                                              trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
            
            return conv_sbbox, conv_mbbox, conv_lbbox
        elif train_backbone==1:
            route_1, route_2,input_data= backbone.darknet19(input_data, self.trainable)
            input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv19')
            input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv20')
            input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv21')
            input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv22')
            input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv23')

            conv_sobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable,'conv_lobj_branch',
                                                    )
            conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 512, 3 * (self.num_class + 9)),
                                              trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv24')
            input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)
            test=conv_sbbox
            with tf.variable_scope('route_1'):
                input_data = tf.concat([input_data, route_2], axis=3)

            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv25')
            input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv26')
            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv27')
            input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv28')
            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv29')

            conv_mobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv_mobj_branch')
            conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (self.num_class + 9)),
                                              trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 128, 64), self.trainable, 'conv30')
            input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

            with tf.variable_scope('route_2'):
                input_data = tf.concat([input_data, route_1], axis=3)

            input_data = common.convolutional(input_data, (1, 1, 128, 64), self.trainable, 'conv31')
            input_data = common.convolutional(input_data, (3, 3, 64, 128), self.trainable, 'conv32')
            input_data = common.convolutional(input_data, (1, 1, 128, 64), self.trainable, 'conv33')
            input_data = common.convolutional(input_data, (3, 3, 64, 128), self.trainable, 'conv34')
            input_data = common.convolutional(input_data, (1, 1, 128, 64), self.trainable, 'conv35')

            conv_lobj_branch = common.convolutional(input_data, (3, 3, 64, 128), self.trainable, 'conv_sobj_branch')
            conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 128, 3 * (self.num_class + 9)),
                                              trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

            return conv_sbbox, conv_mbbox, conv_lbbox
    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 9 + self.num_class))

        conv_raw_head_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_head_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_face_dxdy = conv_output[:, :, :, :, 4:6]
        conv_raw_face_dwdh = conv_output[:, :, :, :, 6:8]
        conv_raw_conf = conv_output[:, :, :, :, 8:9]
        conv_raw_prob = conv_output[:, :, :, :, 9: ]
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])


        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_head_xy = (tf.sigmoid(conv_raw_head_dxdy) + xy_grid) * stride
        pred_head_wh = (tf.exp(conv_raw_head_dwdh) * anchors) * stride
        pred_face_xy = (tf.sigmoid(conv_raw_face_dxdy) + xy_grid) * stride
        pred_face_wh = (tf.exp(conv_raw_face_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_head_xy, pred_head_wh,pred_face_xy,pred_face_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 9 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 8:9]
        conv_raw_prob = conv[:, :, :, :, 9:]

        pred_xywh     = pred[:, :, :, :, 0:8]
        pred_head_xywh= pred_xywh[:,:,:,:,0:4]
        pred_face_xywh= pred_xywh[:,:,:,:,4:8]
        pred_conf     = pred[:, :, :, :, 8:9]

        label_xywh    = label[:, :, :, :, 0:8]
        label_head_xywh=label_xywh[:,:,:,:,0:4]
        label_face_xywh=label_xywh[:,:,:,:,4:8]
        respond_bbox  = label[:, :, :, :, 8:9]
        label_prob    = label[:, :, :, :, 9:]

        giou_head = tf.expand_dims(self.bbox_giou(pred_head_xywh, label_head_xywh), axis=-1)
        giou_face = tf.expand_dims(self.bbox_giou(pred_face_xywh, label_face_xywh), axis=-1)
        giou=(giou_head+giou_face)/2
        input_size = tf.cast(input_size, tf.float32)
        bbox_loss_scale = 2.0 - 1.0 * (label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4]+label_xywh[:, :, :, :, 6:7]*label_xywh[:, :, :, :, 7:8]) /(2* (input_size ** 2))
      
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)
        giou_loss1 =tf.reduce_min(bbox_loss_scale)

        iou_head = self.bbox_iou(pred_head_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, 0:4])
        max_iou_head = tf.expand_dims(tf.reduce_max(iou_head, axis=-1), axis=-1)
        iou_face = self.bbox_iou(pred_face_xywh[:, :, :, :, np.newaxis, :],bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, 4:8])
        max_iou_face = tf.expand_dims(tf.reduce_max(iou_face, axis=-1), axis=-1)
        max_iou=(max_iou_face+max_iou_head)/2

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss



    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss

