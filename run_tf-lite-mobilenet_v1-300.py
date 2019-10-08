#!/usr/bin/env python3

import os
from os import path as osp
import numpy as np
import time
import cv2
import dlr
import datetime
from coco import image_classes
import psutil 
import json
import threading
import copy

ms = lambda: int(round(time.time() * 1000))

dshape = (300,300)
w = 1280
h = 720

def mem_usage():
    process = psutil.Process(os.getpid())
    print("Memory RSS: {:,}".format(process.memory_info().rss))


def transform_image(frame):
    orig_img = frame
    img = cv2.resize(orig_img,dshape)
    img = img.astype('uint8')
    return np.array([img])

demo_path = "/home/aiSage/aws-neo/neo-ai-dlr/demo/aisage"
os.chdir(demo_path)

res = []
model_path, input_tensor_name = "models/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite", "normalized_input_image_tensor"
#modelname, input_tensor_name = "models/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite", "normalized_input_image_tensor"
#model_path = osp.join(demo_path, modelname)
windowName = "Beseye&aiSage"

#cv2.namedWindow(windowName,cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


######################################################################
# Create TVM runtime and do inference

# Build TVM runtime

def getIOU(bbox1, bbox2):
    
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, x1_1, y0_1, y1_1) = bbox1
    (x0_2, x1_2, y0_2, y1_2) = bbox2

    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union 
    
def filt(target,num):
    boxes = []
    output = copy.copy(target)
    if num == 1:
        return output
    for k in range(num):
        boxes.append(target[str(k)]["box"])
    for i in range(num-1):
        for j in range(i+1,num):
            iou = getIOU(boxes[i],boxes[j])
            if iou > 0.3:
                if str(j) in output:
                    del output[str(j)]
    return output


def saveResult(m_out):
    boxes, classes, scores, num_det = m_out
    n_obj = int(num_det[0])
    output = {}
    idx = 0
    for i in range(n_obj):
        res = {}
        cl_id = int(classes[0][i])+1
        score = scores[0][i]
        if cl_id != 1 or score < 0.6:
            continue
        label = "Standing person"
        box = boxes[0][i]
        tbox = [box[1] * w, box[3] * w,box[0] * h, box[2] * h]
        res['label'] = label
        res['box'] = tbox
        res["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output[str(idx)] = res
        idx+=1
    result_box = filt(output,len(output))
    if len(result_box) == 0:
        return False
    timeStamp = datetime.datetime.now().strftime("%S:%f")
    targetPath = "/home/aiSage/aws-neo/neo-ai-dlr/demo/sage_chatbot/application/res"
    jsonFileName = timeStamp+"-result.json"
    jsonFilePath = osp.join(targetPath,jsonFileName)
    with open(jsonFilePath, 'w') as f:
        json.dump(result_box, f)
    return True

def getDetectionSSD(test_img,m):
    inp = transform_image(test_img)
    m_out =  m.run({input_tensor_name:inp})
    boxes, classes, scores, num_det = m_out
    n_obj = int(num_det[0])
    print('-----------')
    for i in range(n_obj):
        cl_id = int(classes[0][i])+1
        if cl_id != 1:
            continue
        label = "Standing person"
        score = scores[0][i]
        box = boxes[0][i]
        if score < 0.6:
            continue
        print(i, label, box, score, datetime.datetime.now())
    print('---end-----')
    return m_out 

def display_300(res, frame, fps, mem):
    outputFps = "fps : {:10.3f}".format(fps)
    cv2.putText(frame, outputFps, (10,40),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0),2,cv2.LINE_AA)
    cv2.putText(frame, mem, (10,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2, cv2.LINE_AA)
    boxes, classes, scores, num_det = res
    n_obj = int(num_det[0])
    scales = [frame.shape[1], frame.shape[0]] * 1
    for i in range(n_obj):
        cl_id = int(classes[0][i])+1
        if cl_id != 1:
            continue
        label = "Standing person"
        score = scores[0][i]
        if score < 0.6:
            continue
        box = boxes[0][i]
        (left, right, top, bottom) = (box[1] * w, box[3] * w,box[0] * h, box[2] * h)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(frame, p1, p2, (221, 245, 66), 3, 1)
        cv2.putText(frame, label, (int(left + 10), int(top+10)), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255, 0, 0), 2, cv2.LINE_AA)

    #cv2.imshow(windowName, frame)

m = dlr.DLRModel(model_path)
cap = cv2.VideoCapture(8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
mbScale = 1024*1124


while True:
    ret, test_img = cap.read()
    if(ret == False):
        continue
    start = datetime.datetime.now()
    res = getDetectionSSD(test_img,m)
    end = datetime.datetime.now()
    save = saveResult(res)
    process = psutil.Process(os.getpid())
    mem = "Memory RSS: {:,} MB".format(float(process.memory_info()[0])/mbScale)
    delta = end-start
    fps = 1./float(delta.total_seconds())
    #print(res)
    display_300(res,test_img,fps,mem)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


#cv2.waitKey(0)
#cv2.destroyAllWindows()



