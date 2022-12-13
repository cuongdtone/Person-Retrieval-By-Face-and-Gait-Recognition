# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022

import time

import cv2
import numpy as np

from utils.functions import BuffelessCamera
from models.yolov7 import YOLOv7
from models.fastestdet import FastestDet
from tracker.bot_sort import BoTSORT
from utils.draw import plot_tracking
from utils.graphic_utils import find_faces
from utils.people_tracking import Crowd
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;1"


class Main:
    # cap = BuffelessCamera('rtsp://admin:tpos123456@192.168.1.57/onvif1')
    cap = cv2.VideoCapture('samples/test_3_14h.mp4')
    bev_matrix = np.load('src/M_camera.npy')  # each camera

    detector = YOLOv7(path='src/yolov7-tiny_480x640.onnx', conf_thres=0.6, iou_thres=0.5)
    # detector = FastestDet(path='src/FastestDet.onnx', conf_thres=0.6, iou_thres=0.5)

    tracker = BoTSORT(frame_rate=15)
    crowd = Crowd(bev_matrix)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    fpss = []

    while True:
        # Capture frame-by-frame
        # print('-----------------------')
        st = time.time()
        success, frame = cap.read()  # read the camera frame
        # print('Cap time ', time.time() - st)

        if not success:
            break
        else:
            ### Rec phase
            # st2 = time.time()
            boxes, scores, class_ids = detector(frame)
            # print('Model Time ', time.time() - st2)
            # st2 = time.time()

            detections = []
            for b, s, c in zip(boxes, scores, class_ids):
                if c == 0:
                    detections.append([*b, s, c + 1])
            detections = np.array(detections)
            output_stracks = tracker.update(detections, frame)

            online_tlwhs = []
            online_ids = []
            online_scores = []

            for t in output_stracks:
                tlwh = t.tlwh
                tid = t.track_id
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
            # st2 = time.time()
            people = crowd.update(frame, online_tlwhs, online_ids)
            # st2 = time.time()
            frame, name = plot_tracking(frame, people)

            # print('All time ', time.time() - st)
            cv2.imshow("Detected Objects", frame)
            # Press key q to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
