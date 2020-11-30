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


import serial
# arduino = serial.Serial('COM7', 115200)



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

def get_mid(box):
    x, y, w, h = box
    return int(x + (w / 2))

def ardu(box):
    try:
        mid = get_mid(box)
        print(box,end="   ")
        if mid <= 90:
            print("큰좌회전")
            c = str('e')
            c = c.encode('utf-8')
            # arduino.write(c)
        elif mid <= 220:
            print("좌회전")
            c = str('d')
            c = c.encode('utf-8')
            # arduino.write(c)
        elif mid >= 550:
            print("큰우회전")
            c = str('q')
            c = c.encode('utf-8')
            # arduino.write(c)
        elif mid >= 420:
            print("우회전")
            c = str('a')
            c = c.encode('utf-8')
            # arduino.write(c)
        else:
            print("직진")
            c = str('w')
            c = c.encode('utf-8')
            # arduino.write(c)
    except:
        print('here error')

def draw_outputs(img, outputs, class_names):
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
        ardu(box)
    return img


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file').
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
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

    times = []

    try:
        vid = cv2.VideoCapture(0)   #cam numberq
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    start = 'a'
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        # t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        # t2 = time.time()
        # times.append(t2-t1)
        # times = times[-20:]
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or start == 's':
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            start = 's'
        # img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
        #                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)    #(B,G,R) top tiomer
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if key == ord('x') or start == 'x':
            start = 'x'
        if key == ord('q'):
            break

        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s') or start == 's':
        #     start = 'a'
        #     img, orig_im, dim = prep_image(frame, inp_dim)
        #     cv2.imshow('Frame', frame)
        #     if CUDA:
        #         # im_dim = im_dim.cuda()
        #         img = img.cuda()
        #
        #     with torch.no_grad():
        #         output = model(Variable(img), CUDA)
        #     output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
        #     try:
        #         if output[0][0].tolist() == 0:  # tensor[0][0] mean the category of predicted class
        #             # and 0 is person in coco.names
        #             initBB = np.array([int(i) for i in output[0][1:5].tolist()])
        #             x, y, w, h = initBB
        #             initBB = (x, y, w, h)
        #             frame_user = frame[x - 10:w + 10, y - 10:h + 10]
        #             cv2.imwrite('./user_faces/user.jpg', frame_user)
        #             tracker.init(frame, initBB)
        #             fps = FPS().start()
        #     except:
        #         print("다시해주세요")


    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
