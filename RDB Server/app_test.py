# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022

import json
import sqlite3
import time

from flask import Flask, request, jsonify
from pathlib import Path
import base64
import numpy as np
import cv2

from utils.face_recognize_threading import FaceRecManager
from utils.load_data import load_user_data

data, face_tree, _ = load_user_data()
face_rec = FaceRecManager(data=data, face_tree=face_tree)

app = Flask(__name__)


@app.route('/api/face_recognize', methods=['POST'])
def api_face_recognize():
    data = request.get_json()
    image_code = eval(data['img'])
    nparr = np.fromstring(image_code, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    info = face_rec.get(data['device_id'], data['track_id'], img)
    return jsonify(info)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
    # app.run()
