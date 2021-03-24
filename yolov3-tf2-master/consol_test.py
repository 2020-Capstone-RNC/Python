import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
import numpy as np
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
# from yolov3_tf2.utils import draw_outputs

import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import time


# from libs.preprocess import prep_image, inp_to_image, letterbox_image
# from libs.darknet import Darknet
# from darknet import Darknet
# import sys, os
# sys.path.append(os.path.join(os.getcwd(),'darkflow/dark/'))
# sys.path.append(os.getcwd().replace('darknet',''))
# import darknet

import serial
import threading

#arduino = serial.Serial('COM3', 115200)

start = 'a'
def arduino_read():
    while True:
        data = arduino.read().decode()
        global start
        start = str(data)
        # print(data)

def ardu_stop():
    for i in range(20):
        print('정지')
        c = str('x')
        c = c.encode('utf-8')
        #arduino.write(c)

def ardu_detect():
    for i in range(20):
        print('찾는중')
        c = str('t')
        c = c.encode('utf-8')
        #arduino.write(c)

def get_mid(box):   #box x좌표 중앙값 추출
    x, y, w, h = box
    return int(x + (w / 2))

def ardu(box):      #중앙값에 따른 카트 방향제어
    try:
        mid = get_mid(box)
        print(box, end="   ")
        if mid <= 90:
            print("큰좌회전")
            c = str('e')
            c = c.encode('utf-8')
            #arduino.write(c)
        elif mid <= 220:
            print("좌회전")
            c = str('d')
            c = c.encode('utf-8')
            #arduino.write(c)
        elif mid >= 550:
            print("큰우회전")
            c = str('q')
            c = c.encode('utf-8')
            #arduino.write(c)
        elif mid >= 420:
            print("우회전")
            c = str('a')
            c = c.encode('utf-8')
            #arduino.write(c)
        else:
            print("직진")
            c = str('w')
            c = c.encode('utf-8')
            #arduino.write(c)
        # time.sleep(2)
    except:
        print('here error')


def draw_outputs(img, outputs, class_names):    #box추출
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        if(class_names[int(classes[i])]) != 'person':   #person만 인식
            continue
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 255, 0), 2)
        # img = cv2.putText(img, '{} {:.4f}'.format(
        #     class_names[int(classes[i])], objectness[i]),
        #     x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
        box = [x1y1[0], x1y1[1], x2y2[0] - x1y1[0], x2y2[1] - x1y1[1]]
        # ardu(box)
    return img

def start_tracker(img,outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        if (class_names[int(classes[i])]) != 'person':  # person만 인식
            continue
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 255, 0), 2)
        # img = cv2.putText(img, '{} {:.4f}'.format(
        #     class_names[int(classes[i])], objectness[i]),
        #     x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
        box = [x1y1[0], x1y1[1], x2y2[0] - x1y1[0], x2y2[1] - x1y1[1]]
        ardu(box)
        return box
    return None

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path t:o video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('cfgfile', './data/yolov3.cfg', 'Config file')


def main(_argv):
    th = threading.Thread(target=arduino_read)
    th.start()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    CUDA = torch.cuda.is_available()
    times = []

    inp_dim = int('416')
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    try:
        vid = cv2.VideoCapture(0)   #cam number - usb=1     vid = cap
    except:
        vid = cv2.VideoCapture(FLAGS.video)
    # vid = cv2.VideoCapture(0)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    tracker = cv2.TrackerKCF_create()
    fps = None
    initBB = None
    # initBB = True
    redetect = False
    failCnt = 0
    global start
    check_start = None

    while True:
        ret, frame = vid.read() #img = frame
        # if frmae is None:
        #     logging.warning("Empty Frame")
        #     time.sleep(0.1)
        #     continue
        # frame = imutils.resize(frame, width = 500)
        (H, W) = frame.shape[:2]

        img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)
        boxes, scores, classes, nums = yolo.predict(img_in)

        # t1 = time.time()
        # boxes, scores, classes, nums = yolo.predict(img_in)
        # t2 = time.time()
        # times.append(t2-t1)
        # times = times[]




        # print('initBB = ', initBB)
        if initBB is not None:
            (success, box) = tracker.update(frame)
            if start == 'q':
                initBB = None
                tracker = cv2.TrackerMedianFlow_create()
                ardu_stop()
            elif success:
                failCnt = 0
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x+30, y), (x + w-30, y + h), (0, 255, 0), 2)
                ardu(box)
                # frame = draw_outputs(frame, (boxes, scores, classes, nums), class_names)
            else:
                failCnt += 1
                ardu_detect()
                if failCnt > 50:
                    redetect = True
                    initBB = None
                    tracker = cv2.TrackerKCF_create()
            # fps.update()
            # fps.stop()
            # info = [
            #     ('Tracker', 'kcf'),
            #     ('Success', 'yes' if success else 'No'),
            #     ('FPS', '{:.2f}'.format(fps.fps())),
            # ]
            #
            # for (i, (k, v)) in enumerate(info):
            #     text = '{}:{}'.format(k, v)
            #     cv2.putText(frame, text, (10, H-((i*20)+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255, 2))

        # if redetect:
        #     ret, frame = vid.read()
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     template = cv2.imread('./user_faces/user.jpg', 0)
        #     w, h = template.shape[::-1]
        #
        #     res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF)
        #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #     top_left = min_loc
        #     bottom_right = (top_left[0] + w, top_left[1] + h)
        #     cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)

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
        if key == ord('s') or start == 's':
            check_start = start_tracker(frame, (boxes, scores, classes, nums), class_names)
            if check_start is not None:
                start = 'a'
                initBB = tuple(check_start)
                tracker.init(frame, initBB)
                x, y, w, h = check_start
                frame_user = frame[y:y+h,x+20:x+w-20]
                cv2.imwrite('./user_faces/user.jpg', frame_user)
                # fps = FPS().start()

            # img, orig_im, dim = prep_image(frame, inp_dim)
            # img = prep_image(frame, inp_dim)

            # cv2.imshow('img', frame)

            # if CUDA:
            #     # im_dim = im_dim.cuda()
            #     img = img.cuda()
            #
            # with torch.no_grad():
            #     output = model(Variable(img),CUDA)
            # output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            # try:
            #     if output[0][0].tolist() == 0:
            #         #tensor[0][0] mean the category of predicted class and 0 is person in coco.names
            #         initBB = np.array([int(i) for i in output[0][1:5].tolist])
            #         x, y, w, h = initBB
            #         initBB = (x, y, w, h)
            #         frame_user = frame[x-10:w+10, y-10:h+10]
            #         cv2.imwrite('./user_faces/user.jpg', frame_user)
            #         tracker.init(frame, initBB)
            #         fps = FPS().start()
            # except:
            #     print("다시해주세요")
        #print(initBB)
        cv2.imshow('img', frame)

        if key == ord('q'):
            ardu_stop()
            break

    vid.release()

    cv2.destroyAllWindows()



        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s') or start == 's':
        #     frame = draw_outputs(frame, (boxes, scores, classes, nums), class_names)
        #     start = 's'
        # # img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
        # #                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)    #(B,G,R) top tiomer
        # if FLAGS.output:
        #     out.write(frame)
        # cv2.imshow('output', frame)
        # if key == ord('x') or start == 'x':
        #     start = 'x'
        # if key == ord('q'):
        #     break







if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
