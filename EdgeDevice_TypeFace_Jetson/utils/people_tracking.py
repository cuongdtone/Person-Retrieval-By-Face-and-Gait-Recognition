# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022
import os.path
import time

import cv2
from models.face_recognizer import face_compare_tree
from models.face_aligner import norm_crop
import base64
from unidecode import unidecode
from datetime import datetime
from utils.api import insert_timekeeping_to_server
from utils.device_control import LinearDevice
import json
import random
import glob

device = LinearDevice()
device.start()
skip_dict = {}

with open('src/format_audio.json', 'rb') as f:
    audio_data = json.loads(f.read())


def get_audio(time_now, code):
    audio_type = []
    for folder, audio in audio_data.items():
        times = audio['time']
        for t in times:
            if t[0] < time_now < t[1]:
                audio_type.append(folder)
    if len(audio_type) == 0:
        audio_type.append('hello')
    audio_type = random.choice(audio_type)
    file = glob.glob(f'media/{audio_type}/{code}_*.wav')
    if len(file) != 0:
        return random.choice(file)
    return None


class Face(object):
    def __init__(self, track_id):
        self.id = track_id
        self.name = None
        self.id_code = None
        self.image = None
        self.mask = True
        self.fas = True
        self.last_time = time.time()
        self.sim_threshold = 0.45
        self.count_fas = 0
        self.fas_allow_time = 2 * 25  # 1 second
        self.hour_skip_timekeep = 10  # second
        self.off_shift_hour = 16.5  # 16h30p
        self.area_ratio = 0.015
        self.error_align_face = 34.5  # 1 - 112, norm is 35.5

    def update(self, image, face_box, kps, face_recognition, search_tree, employees_data, face_mask, face_fas):
        image = image.copy()
        h, w = image.shape[:2]
        self.last_time = time.time()
        if self.count_fas > self.fas_allow_time:
            return
        if self.name is None:
            aimg, d_e = norm_crop(image.copy(), kps, return_d_eye=True)
            if d_e < self.error_align_face:
                return
            h_face, w_face = face_box[3] - face_box[1], face_box[2] - face_box[0]

            if (h_face * w_face) / (h * w) < self.area_ratio:
                return

            feat = face_recognition.get(aimg)
            info = face_compare_tree(feat, employees_data, search_tree)
            if info['fullname'] != 'uknown' and info['Sim'] > self.sim_threshold:
                self.name = info['fullname']
                self.id_code = info['code']
                self.image = image
                hour = datetime.now().hour
                minute = datetime.now().minute / 60
                time_now = float(hour) + float(minute)
                if self.name not in skip_dict.keys() or \
                        (datetime.now() - skip_dict[
                            self.name]).total_seconds() >= self.hour_skip_timekeep:

                    insert_timekeeping_to_server(self.id_code, self.name)
                    skip_dict.update({self.name: datetime.now()})
                    path = get_audio(time_now, self.id_code)
                    path = os.path.join(os.getcwd(), path)
                    # print(path)
                    # print(os.path.exists(path))
                    device.queue.append(os.path.abspath(path))


class Crowd(object):
    def __init__(self):
        self.people = {}  # contain {id: person}
        self.ids = []
        self.delete_person_delay = 1  # tracking will cancel in 3s

    def update(self, frame, bboxes, ids, kpss, face_recognition, search_tree, employees_data, face_mask, face_fas):
        # people is dict: key is tracking id, value is [facebox]
        result_track = {}
        for track_id, face_box, kps in zip(ids, bboxes, kpss):

            if track_id not in self.people.keys():
                self.people.update({track_id: Face(track_id)})

            self.people[track_id].update(frame, face_box, kps,
                                         face_recognition, search_tree, employees_data, face_mask, face_fas)

        self.delete_tracking()
        return result_track

    def delete_tracking(self):
        now = time.time()
        keys = self.people.keys()
        for i in keys:
            person = self.people[i]
            if now - person.last_time > self.delete_person_delay:
                self.people = self.remove_key(self.people, i)

    @staticmethod
    def remove_key(d, key):
        r = dict(d)
        del r[key]
        return r
