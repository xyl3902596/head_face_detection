#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict
#Easydict的作用是生成一个字典C，可以像访问类的属性一样访问这个字典的属性，普通的字典需要C['XXX']，Easydict可以C.XXX

__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "./data/classes/coco.names"
__C.YOLO.ANCHORS                = "./data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.UPSAMPLE_METHOD        = "deconv"
__C.YOLO.IOU_LOSS_THRESH        = 0.5
# Train options
__C.TRAIN                       = edict()
__C.TRAIN.ANNOT_PATH            = "/home/yckj2032/HeadFaceDetectionData/train_convert_label.txt"
__C.TRAIN.BATCH_SIZE            = 6
__C.TRAIN.INPUT_SIZE            = [544]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 40
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_test_loss=9.0348.ckpt-34"
__C.TRAIN.INITIAL_WEIGHT_RESTORE= "./checkpoint/yolov3_test_loss=9.0348.ckpt-34"


# TEST options
__C.TEST                        = edict()
__C.TEST.ANNOT_PATH             = "/home/yckj2032/HeadFaceDetectionData/val_convert_label.txt"
__C.TEST.BATCH_SIZE             = 6
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = False
__C.TEST.WEIGHT_FILE            ="./checkpoint/yolov3_test_loss=9.0348.ckpt-34"
__C.TEST.WEIGHT_FILE_RESTORE    = "./checkpoint/yolov3_test_loss=4.0366.ckpt-26"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.38
__C.TEST.IOU_THRESHOLD          = 0.3
__C.TEST.TEST_VIDEO_PATH        = '/home/yckj2032/video/metro_outdoor_front_midrange_overlook_1_2.mp4'
__C.TEST.PREDICT_VIDEO_PATH     = '/home/yckj2032/video/predict/metro_outdoor_front_midrange_overlook_1_2_perdict.mp4'



