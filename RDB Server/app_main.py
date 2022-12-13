# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022

import cv2
import numpy as np
from flask import Flask, request, jsonify

from utils.gait_recognizer_threading import GaitRecManager
from utils.load_data import load_user_data
from utils.sqlite_db import get_all_people, insert_timekeeping, get_all_people_web, get_timekeep, insert_new_person

app = Flask('RDB Server')
data, face_tree, gait_tree = load_user_data()
gait_rec = GaitRecManager(num_threads=8, data=data, gait_tree=gait_tree)


@app.route('/api/face_recognize', methods=['GET'])
def api_face_recognize():
    users_info = get_all_people()
    response = {'data': users_info}
    return jsonify(response)


@app.route('/api/insert_face_log', methods=['GET'])
def insert_face_log():
    data_receive = request.get_json()
    insert_timekeeping(data_receive['code'], data_receive['fullname'], data_receive['device'])
    return jsonify(True)


@app.route('/api/gait_recognize', methods=['POST'])
def api_gait_recognize():
    data_receive = request.get_json()
    clip_code = eval(data_receive['clip'])
    clip = []

    for img_code in clip_code:
        img_code = eval(img_code)
        nparr = np.fromstring(img_code, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        clip.append(img)
    info = gait_rec.get(data_receive['device'], data_receive['track_id'], clip, data, gait_tree)
    print(info)
    if info['sim'] > 0.45:
        insert_timekeeping(info['code'], info['fullname'], data_receive['device'])
        info.update({'checked': True})
        return jsonify(info)
    else:
        return jsonify({'checked': False})


### Database Server
@app.route('/api/get_people_table', methods=['GET'])
def get_people_table():
    info = get_all_people_web()
    return jsonify({'data': info})


@app.route('/api/get_timekeep', methods=['GET'])
def get_timekeep_f():
    request_data = request.get_json()
    info = get_timekeep(now_month=request_data['month'], year=request_data['year'])
    return jsonify({'data': info})


@app.route('/api/add_person_to_db', methods=['POST'])
def add_person():
    request_data = request.get_json()
    insert_new_person(request_data['code'], request_data['fullname'], request_data['vocative'],
                      request_data['birthday'], request_data['branch'], request_data['position'])

    return jsonify(True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
    # app.run()
