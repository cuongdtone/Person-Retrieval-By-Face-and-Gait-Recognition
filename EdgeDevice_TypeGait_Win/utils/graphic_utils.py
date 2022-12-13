# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/29/2022

import cv2
import numpy as np


def compute_inside_box(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    iou = interArea / boxAArea
    # return the intersection over union value
    return iou


def find_faces(tlwhs_body, ids):
    info = {}
    for body_box, object_id in zip(tlwhs_body, ids):
        x1, y1, w, h = body_box
        int_body_xyxy = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        face_box = None
        kps = None
        info.update({object_id: {'body': int_body_xyxy, 'face': face_box, 'kps': kps}})
    return info


def warp_point(x: int, y: int, M):
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    return (
        int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
        int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
    )


def resize(image, window_height=300):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    window_width = window_height / aspect_ratio
    image = cv2.resize(image, (int(window_height), int(window_width)))
    return image
