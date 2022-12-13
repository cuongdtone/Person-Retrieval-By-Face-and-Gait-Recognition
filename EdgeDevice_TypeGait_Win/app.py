# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022

import time

import cv2
import numpy as np
import yaml
from flask import Flask, render_template, Response

from models.yolov7 import YOLOv7
from models.fastestdet import FastestDet
from tracker.bot_sort import BoTSORT
from utils.draw import plot_tracking
from utils.graphic_utils import find_faces
from utils.people_tracking import Crowd
from threading import Thread

app = Flask("Face Edge Device")

# Initial
with open("settings.yaml", 'r') as f:
    cfg = yaml.safe_load(f)
cap = cv2.VideoCapture(cfg['source']) #, cv2.CAP_DSHOW)
bev_matrix = np.load('src/M_camera.npy')  # each camera

# detector = YOLOv7(path='src/yolov7-tiny_256x320.onnx', conf_thres=0.6, iou_thres=0.5)
detector = FastestDet(path='src/FastestDet.onnx', conf_thres=0.6, iou_thres=0.5)

tracker = BoTSORT(frame_rate=15)
crowd = Crowd(bev_matrix)

frame = None
run_flag = False
byte_frame = None
fps = 0
name = 'none'


def rec():
    global run_flag, byte_frame, fps, name, frame
    if run_flag:
        while True:
            yield byte_frame
            time.sleep(0.001)
    else:
        run_flag = True
        while True:
            # Capture frame-by-frame
            st = time.time()
            success, frame_ori = cap.read()  # read the camera frame
            if not success:
                break
            else:
                ### Rec phase

                boxes, scores, class_ids = detector(frame_ori)
                detections = []
                for b, s, c in zip(boxes, scores, class_ids):
                    if c == 0:
                        detections.append([*b, s, c + 1])
                detections = np.array(detections)

                output_stracks = tracker.update(detections, frame_ori)

                online_tlwhs = []
                online_ids = []
                online_scores = []

                for t in output_stracks:
                    tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                people = crowd.update(frame_ori, online_tlwhs, online_ids)
                frame, name = plot_tracking(frame_ori, people)
                fps = 1 / (time.time() - st)

                ### Phase 2
                if byte_frame is None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    byte_frame = (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                yield byte_frame


def gen_frames():
    global byte_frame, frame
    while True:
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_b = buffer.tobytes()
            byte_frame = (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_b + b'\r\n')
        except:
            pass


@app.route('/video_feed')
def video_feed():
    return Response(rec(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fps_process')
def fps_process():
    def generate():
        yield '%.2f' % fps
    return Response(generate(), mimetype='text')


@app.route('/name_show', methods=['GET'])
def name_show():
    def generate():
        yield name
    return Response(generate())


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', device_id=cfg['device_id'])


if __name__ == '__main__':
    Thread(target=gen_frames).start()
    app.run(host='0.0.0.0', debug=False)
