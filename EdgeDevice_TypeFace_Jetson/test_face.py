# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 11/4/2022
import time

import cv2
from models.face_recognizer import ArcFaceONNX

rec = ArcFaceONNX('src/w600k_mbf.onnx')

image = cv2.imread('src/test.png')
image = cv2.resize(image, (112, 112))
st = time.time()
feat = rec.get(image)
print('Rec Time: ', time.time() - st)
print(feat)
