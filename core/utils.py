#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:14:19
#   Description :
#
#================================================================

import cv2
import random
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)
def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):
  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/float(w), ih/float(h))
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2,4,6]] = gt_boxes[:, [0, 2,4,6]] * scale + dw
        gt_boxes[:, [1, 3,5,7]] = gt_boxes[:, [1, 3,5,7]] * scale + dh
        return image_paded, gt_boxes
  

def draw_bbox(image, bboxes,head_or_face, classes=['front','rear']):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    image_h, image_w, _ = image.shape
    class_convert=['head','face']
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:8], dtype=np.int32)
        fontScale = 0.6
        score = bbox[8]
        class_ind = int(bbox[9])
             
        bbox_thick = int(1.0 * (image_h + image_w) / 600)
        c_head_min, c_head_max = (coor[0], coor[1]), (coor[2], coor[3])
        c_face_min, c_face_max = (coor[4], coor[5]), (coor[6],coor[7])

        if class_ind==0:
            if head_or_face=="head_and_face" or head_or_face=="head":
                cv2.rectangle(image, c_head_min, c_head_max, [0, 0, 255], bbox_thick)
                bbox_mess_head = '%s: %.2f' % (class_convert[0], score)
                cv2.putText(image, bbox_mess_head, (c_head_min[0], c_head_min[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 255), bbox_thick, lineType=cv2.LINE_AA)
            if head_or_face=="head_and_face" or head_or_face=="face":
                cv2.rectangle(image, c_face_min, c_face_max, [0, 255, 0], bbox_thick)
                bbox_mess_face = '%s: %.2f' % (class_convert[1], score)
                cv2.putText(image, bbox_mess_face, (c_face_min[0], c_face_min[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 255, 0), bbox_thick , lineType=cv2.LINE_AA)
        if class_ind == 1:
            if head_or_face=="head_and_face" or head_or_face=="head":
                cv2.rectangle(image, c_head_min, c_head_max, [0, 255, 0], bbox_thick)
                bbox_mess_head = '%s: %.2f' % (class_convert[0], score)
                cv2.putText(image, bbox_mess_head, (c_head_min[0], c_head_min[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 255), bbox_thick, lineType=cv2.LINE_AA)

    return image



def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious



def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
   
    classes_in_img = list(set(bboxes[:, 9]))
    best_bboxes = []

   

    while len(bboxes) > 0:
            max_ind = np.argmax(bboxes[:, 8])
            best_bbox = bboxes[max_ind]
            best_bboxes.append(best_bbox)
            bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
         
            iou=bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            bboxes[:, 8] = bboxes[:, 8] * weight
            score_mask = bboxes[:, 8] > 0.
            bboxes = bboxes[score_mask]
            
    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
     
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)
    pred_head_xywh = pred_bbox[:, 0:4]
    pred_face_xywh = pred_bbox[:, 4:8]
    pred_conf = pred_bbox[:, 8]
    pred_prob = pred_bbox[:, 9:]
    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_head_coor = np.concatenate([pred_head_xywh[:, :2] - pred_head_xywh[:, 2:] * 0.5,
                                pred_head_xywh[:, :2] + pred_head_xywh[:, 2:] * 0.5], axis=-1)
    pred_face_coor = np.concatenate([pred_face_xywh[:, :2] - pred_face_xywh[:, 2:] * 0.5,
                                     pred_face_xywh[:, :2] + pred_face_xywh[:, 2:] * 0.5], axis=-1)
   
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / float(org_w), input_size / float(org_h))

    dw = (input_size - resize_ratio * org_w) / 2.0
    dh = (input_size - resize_ratio * org_h) / 2.0

    pred_head_coor[:, 0::2] = 1.0 * (pred_head_coor[:, 0::2] - dw) / resize_ratio
    pred_head_coor[:, 1::2] = 1.0 * (pred_head_coor[:, 1::2] - dh) / resize_ratio
    pred_face_coor[:, 0::2] = 1.0 * (pred_face_coor[:, 0::2] - dw) / resize_ratio
    pred_face_coor[:, 1::2] = 1.0 * (pred_face_coor[:, 1::2] - dh) / resize_ratio
    
    
    pred_head_coor = np.concatenate([np.maximum(pred_head_coor[:, :2], [0, 0]),
                                np.minimum(pred_head_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    
    invalid_mask = np.logical_or((pred_head_coor[:, 0] > pred_head_coor[:, 2]), (pred_head_coor[:, 1] > pred_head_coor[:, 3]))
    pred_head_coor[invalid_mask] = 0
 
    pred_face_coor = np.concatenate([np.maximum(pred_face_coor[:, :2], [0, 0]),
                                np.minimum(pred_face_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_face_coor[:, 0] > pred_face_coor[:, 2]), (pred_face_coor[:, 1] > pred_face_coor[:, 3]))
    pred_face_coor[invalid_mask] = 0
  
    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_head_coor[:, 2:4] - pred_head_coor[:, 0:2], axis=-1))
    scale_mask_head = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_face_coor[:, 2:4] - pred_face_coor[:, 0:2], axis=-1))
    scale_mask_face = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
 
    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_head_coor)), classes]
    score_mask = scores> score_threshold
  
    mask_head = np.logical_and(scale_mask_head, score_mask)
    mask_face = np.logical_and(scale_mask_face, score_mask)
    mask=np.logical_and(mask_head,mask_face)
    coors_head,coors_face,scores, classes = pred_head_coor[mask],pred_face_coor[mask],scores[mask], classes[mask]
   
    

    return np.concatenate([coors_head, coors_face,scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)



