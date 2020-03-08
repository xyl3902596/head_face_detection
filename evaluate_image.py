#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os

import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3
import argparse
from tensorflow.python import pywrap_tensorflow
#np.set_printoptions(threshold=np.inf)
class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = {0:'front',1:'rear'}
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.test_video_path  = cfg.TEST.TEST_VIDEO_PATH
        self.predict_video_path= cfg.TEST.PREDICT_VIDEO_PATH

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32,shape=[1,544,544,3], name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')
        
        model =YOLOV3(self.input_data, self.trainable,1)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox
        self.conv_sbbox, self.conv_mbbox, self.conv_lbbox = model.conv_sbbox, model.conv_mbbox, model.conv_lbbox
        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)


        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weight_file)
      
    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])

        image_data = image_data[np.newaxis, ...]
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        pred_sbbox, pred_mbbox, pred_lbbox,conv_sbbox,conv_mbbox,conv_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox,self.conv_sbbox, self.conv_mbbox, self.conv_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            },options=options, run_metadata=run_metadata)

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 9 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 9 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 9 + self.num_classes))], axis=0)
        

        bboxes1 = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes1, self.iou_threshold)

        return bboxes


    def evaluate(self,head_or_face):
        result_dir_path = './image/result'
        test_dir_path = './image/test'
        for test_pic_path in os.listdir(test_dir_path):
            test_pic_full_path=os.path.join(test_dir_path,test_pic_path)
            image = cv2.imread(test_pic_full_path)
            print('=> pic %s:' % test_pic_path)
            bboxes_pr= self.predict(image)

            if self.write_image:
                image = utils.draw_bbox(image, bboxes_pr,head_or_face=head_or_face)
                cv2.imwrite(os.path.join(result_dir_path,test_pic_path), image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--head_or_face', type=str, help="detect head or face or both", default='head_and_face')
    args = parser.parse_args()
    print args.head_or_face
    YoloTest().evaluate(head_or_face=args.head_or_face)
