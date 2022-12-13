# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022

import time
import requests
import cv2
import base64
import json
import yaml


with open("settings.yaml", 'r') as f:
    cfg = yaml.safe_load(f)


def sent_request_face(image, device_id, track_id):
    url = f'{cfg["rdb_server"]}/api/face_recognize'
    _, img_encoded = cv2.imencode('.jpg', image)
    data = {'device_id': device_id, 'track_id': track_id, 'img': str(img_encoded.tobytes())}
    response = requests.post(url, json=data)
    return response.json()


def sent_request_gait(clip, track_id):
    url = f'{cfg["server"]}/api/gait_recognize'
    bytes_vid = []
    for image in clip:
        _, img_encoded = cv2.imencode('.jpg', image)
        bytes_vid.append(str(img_encoded.tobytes()))
    data = {'device': cfg['device'], 'track_id': track_id, 'clip': str(bytes_vid)}
    response = requests.post(url, json=data)
    return response.json()
