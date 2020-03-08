#coding=utf-8
import numpy as np
def annotation_convert(label_path,label_convert_path):
    with open(label_path,'r') as fr:
        with open(label_convert_path,'a') as fw:
            txt=fr.readlines()
            annotation=[line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            for line in annotation:
                bbox=line.split()[1:]
                image_path=line.split()[0]
                bboxes = [list(map(int, box.split(','))) for box in bbox]
                convert_bboxes=label_convert(bboxes)
                anno_convert=image_path
                for convert_box in convert_bboxes:
                    anno_convert+=' '+','.join([str(x) for x in convert_box[0]+convert_box[1]])+','+str(convert_box[2])
                fw.write(anno_convert+'\n')

def list_average(x,y):
    avergae=[]
    for i in range(len(x)):
        avergae.append((x[i]+y[i])/2)
    return avergae
def label_convert(bboxes,ratio=0.15):
     head_boxes=[ box for box in bboxes if box[-1]==0]
     face_boxes=[ box for box in bboxes if box[-1]==1]
     convert_box=[]
     for head_box in head_boxes:
         find_flag=False
         head_width=head_box[2]-head_box[0]
         head_height=head_box[3]-head_box[1]
         for face_box in face_boxes:
            if  face_box[0]>head_box[2]-head_width*(1+ratio) and face_box[1]>head_box[3]-head_height*(1+ratio) and face_box[2]<head_box[0]+(1+ratio)*head_width and face_box[3]<head_box[1]+(1+ratio)*head_height:
                convert_box.append([head_box[0:4],face_box[0:4],0])#0代表正脸
                find_flag=True
                break
         if not find_flag:
                convert_box.append([head_box[0:4],list_average(head_box[0:2],head_box[2:4])*2,1])#1代表背脸
     return convert_box

label_path=r'F:\Video_Object_detection\HeadFaceDetectionData\train_label.txt'
label_convert_path=r'F:\Video_Object_detection\HeadFaceDetectionData\train_convert_label.txt'
annotation_convert(label_path,label_convert_path)
