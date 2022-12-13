# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/13/2022

import cv2
import numpy as np
import time
from models.face_detecter import RetinaFace
from models.scrfd_face_detector import SCRFD
from models.face_recognizer import ArcFaceONNX
from models.face_mask import MaskDetection
from models.face_fas import FAS
from tracker.bot_sort import BoTSORT
from utils.people_tracking import Crowd
from utils.draw import plot_tracking
from utils.functions import Watcher, BuffelessCamera
from utils.load_data import load_user_data
import yaml
import traceback
from threading import Thread

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;1"


class FaceCameras:
    def __init__(self, config='settings.yaml'):
        self.employees_data = None
        self.search_tree = None
        self.employees_info = None
        self.config = yaml.load(open(config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

        self.cap = BuffelessCamera(self.config['camera'])  # cv2.VideoCapture(self.config['camera'])

        if self.config['detector'] == 'retina':
            self.detector = RetinaFace('src/det_500m.onnx')
        elif self.config['detector'] == 'scrfd_500m':
            self.detector = SCRFD('src/scrfd_500m_bnkps.onnx')
        elif self.config['detector'] == 'scrfd2.5g':
            self.detector = SCRFD('src/scrfd_2.5g_bnkps.onnx')
        else:
            raise
        self.face_analysis = Crowd()
        self.tracker = BoTSORT(self.config['fps_tracker'])
        self.recognizer = ArcFaceONNX('src/w600k_mbf.onnx')
        self.load()
        self.auto_reload()

    def auto_reload(self):
        def reload_():
            while True:
                time.sleep(10)
                self.load()
        Thread(target=reload_, args=()).start()

    def load(self):
        self.employees_data, self.search_tree, self.employees_info = load_user_data()

    def run(self):
        while self.cap.isOpened():
            try:
                st = time.time()
                ret, frame = self.cap.read()
                if ret:
                    faces, kpss = self.detector.detect(frame, input_size=(512, 512))
                    outputs = []
                    for kps, face in zip(kpss, faces):
                        outputs.append([*face, 1])
                    outputs = np.array(outputs)
                    output_stracks = self.tracker.update(outputs, kpss, frame)
                    online_xyxy = []
                    online_ids = []
                    online_scores = []
                    online_kpss = []
                    for t in output_stracks:
                        tlwh = t.tlwh
                        tid = t.track_id
                        x1, y1, w, h = tlwh
                        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                        online_xyxy.append(intbox)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_kpss.append(t.kp)
                    self.face_analysis.update(frame, online_xyxy, online_ids, online_kpss, face_recognition=self.recognizer,
                                              search_tree=self.search_tree, employees_data=self.employees_data,
                                              face_mask=None, face_fas=None
                                              )
                    if self.config['vis']:
                        frame = plot_tracking(frame, online_xyxy, online_ids, online_kpss, frame_id=1, fps=15)
                        cv2.imshow("Detected Objects", frame)
                        #cv2.imwrite(f'images/{c}.jpg', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            except:
                print(traceback.format_exc())
                pass

    @staticmethod
    def sort_kps(boxes, kpss):
        kpss_sorted = np.zeros_like(kpss)
        for i, b in enumerate(boxes):
            for kps in kpss:
                kp1 = kps[0]
                # kp2 = kps[4]
                if b[0] < kp1[0] < b[2] and b[1] < kp1[1] < b[3]:
                    kpss_sorted[i] = kps
                    break
        return kpss_sorted
