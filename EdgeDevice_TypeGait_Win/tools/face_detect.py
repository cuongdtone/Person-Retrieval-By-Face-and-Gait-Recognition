# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022
import cv2

from models.face_detecter import RetinaFaceIR
from models.face_aligner import norm_crop

f = RetinaFaceIR()

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    faces, kpss = f.detect(frame)
    for kp in kpss:
        face, _ = norm_crop(frame, kp)
        print(face)
        cv2.imwrite('../../RDB Server/face.jpg', face)
    cv2.imshow('cc', frame)
    cv2.waitKey(2)
