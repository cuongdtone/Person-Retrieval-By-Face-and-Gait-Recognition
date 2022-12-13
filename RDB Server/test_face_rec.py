# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022
import time

import cv2

from utils.face_recognize_threading import FaceRecManager
from utils.load_data import load_user_data
from threading import Thread

data, face_tree, _ = load_user_data()

face_rec = FaceRecManager(data=data, face_tree=face_tree)

image = cv2.imread('face.jpg')
image2 = cv2.imread('face2.jpg')

st = time.time()
for idx, img in enumerate([image2, image]):
    Thread(target=face_rec.get, args=[idx, 1, img]).start()

print(time.time() - st)

