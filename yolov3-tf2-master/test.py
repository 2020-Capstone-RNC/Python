from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from libs.util import *
from libs.darknet import Darknet
from libs.preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
import socket

import imutils
from imutils.video import VideoStream
from imutils.video import FPS

import serial
import threading
import time

arduino = serial.Serial('COM7', 115200)
start = 'a'

def arduino_read():
    while True:
        data = arduino.read().decode()
        global start
        start = str(data)
        # print(data)

def ardu_stop():
    for i in range(20):
        # print('정지')
        c = str('x')
        c = c.encode('utf-8')
        arduino.write(c)

def ardu_detect():
    for i in range(20):
        # print('찾는중')
        c = str('t')
        c = c.encode('utf-8')
        arduino.write(c)


def ardu(boxs):
    try:
        (x, y, w, h) = [int(v) for v in boxs]
        mid = int(x + (w / 2))
        if mid <= 100:
            # print("큰좌회전")
            c = str('e')
            c = c.encode('utf-8')
            arduino.write(c)
        elif mid <= 200:
            # print("좌회전")
            c = str('d')
            c = c.encode('utf-8')
            arduino.write(c)
        elif mid >= 400:
            # print("큰우회전")
            c = str('q')
            c = c.encode('utf-8')
            arduino.write(c)
        elif mid >= 300:
            # print("우회전")
            c = str('a')
            c = c.encode('utf-8')
            arduino.write(c)
        else:
            # print("직진")
            c = str('w')
            c = c.encode('utf-8')
            arduino.write(c)
    except:
        print('here error')


def get_largest_box(boxs):
    largest = None
    for box in boxs:
        if box == None:
            continue
        elif largest == None:
            largest = box
        else:
            largestS = largest[2] * largest[3]
            boxS = box[2] * box[3]
            if (largestS < boxS):
                largest = box
    return largest

def write(x, img):
    c1 = (int(x[1]), int(x[2]))
    c2 = (int(x[3]), int(x[4]))
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label != "person":
        return
    box = (c1[0], c1[1], c2[0] - c1[0], c2[1] - c1[1])
    cv2.rectangle(img, c1, c2, (0, 255, 0), 1)
    return box


def arg_parse():
    parser = argparse.ArgumentParser(
        description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help="Video to run detection upon",
                        default="videos/video.mp4", type=str)
    parser.add_argument("--dataset", dest="dataset",
                        help="Dataset on which the network has been trained", default="pascal")
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="config/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument('-t', '--tracker', type=str, default='kcf',
                        help='OpenCV object tracker type')
    return parser.parse_args()


if __name__ == '__main__':
    th = threading.Thread(target=arduino_read)
    th.start()

    ################# yolo set #########################
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    CUDA = torch.cuda.is_available()
    num_classes = 80
    bbox_attrs = 5 + num_classes

    # print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    # print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()
    model.eval()

    classes = load_classes('data/coco.names')
    ################# tracker set #########################
    tracker = cv2.TrackerKCF_create()

    fps = None
    initBB = None
    #cap = cv2.VideoCapture('rtsp://192.168.0.210:8554/test')
    cap = cv2.VideoCapture(1)

    redetect = False
    failCnt = 0
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]
        if initBB is not None:
            (success, box) = tracker.update(frame)
            if start == 'q' :
                initBB = None
                tracker = cv2.TrackerMedianFlow_create()
                ardu_stop()
            elif success :
                failCnt = 0
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # print("[",x,y,w,h,"]")
                ardu(box)
            else :
                failCnt += 1
                ardu_detect()
                if failCnt > 50:
                    redetect = True
                    initBB = None
                    tracker = cv2.TrackerKCF_create()
            fps.update()
            fps.stop()
            info = [
                ('Tracker', 'kcf'),
                ('Success', 'yes' if success else 'No'),
                ('FPS', '{:.2f}'.format(fps.fps())),
            ]

            for (i, (k, v)) in enumerate(info):
                text = '{}: {}'.format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255, 2))

        if redetect:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template = cv2.imread('./user_faces/user.jpg', 0)
            w, h = template.shape[::-1]

            res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 1)

            """
            imgray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            w, h = imgray.shape[::-1]
            templ = cv2.imread('./user.jpg', cv2.IMREAD_GRAYSCALE)
            templ_h, templ_w = templ.shape[::-1]
            res = cv2.matchTemplate(imgray, templ, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.6)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
            """

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or start=='s':
            start = 'a'
            img, orig_im, dim = prep_image(frame, inp_dim)
            cv2.imshow('Frame', frame)
            if CUDA:
                # im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            try :
                if output[0][0].tolist() == 0: # tensor[0][0] mean the category of predicted class
                    # and 0 is person in coco.names
                    initBB = np.array([int(i) for i in output[0][1:5].tolist()])
                    x, y, w, h = initBB
                    initBB = (x, y, w, h)
                    frame_user = frame[x-10:w+10,y-10:h+10]
                    cv2.imwrite('./user_faces/user.jpg', frame_user)
                    tracker.init(frame, initBB)
                    fps = FPS().start()
            except:
                print("다시해주세요")
        # print(initBB)
        cv2.imshow('Frame', frame)

        if key == ord('q'):
            ardu_stop()
            break

    cap.release()

    cv2.destroyAllWindows()