# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/20/2022
import os.path
import time
from threading import Thread
import yaml
import cv2
import numpy as np

from utils.api import sent_request_gait
from utils.draw import plot
from utils.graphic_utils import warp_point

skip_dict = {}

with open("settings.yaml", 'r') as f:
    cfg = yaml.safe_load(f)


class Person(object):
    def __init__(self, track_id):
        self.id = track_id
        self.name = None
        self.id_code = None
        self.last_time = time.time()
        self.vis = False
        self.requested_flag = False
        self.save_clip = False

        # gait extract
        self.bev_map_positions = []
        self.frames = []
        self.walking_clip = None
        self.walking_distance = cfg['walking_distance']
        self.gait_threshold = cfg['gait_threshold']

    def update(self, image, body, bev):
        image = image.copy()
        self.last_time = time.time()

        if self.name is None:  # and not self.have_recognize_data:
            # Phase 2: get walking clip
            x1, y1, x2, y2 = body
            body_ground_point = [int((x1 + x2) / 2), y2]
            position_in_bev_map = warp_point(*body_ground_point, bev)
            self.bev_map_positions.append(position_in_bev_map)
            self.frames.append(image[body[1]: body[3], body[0]:body[2], :])

            if len(self.bev_map_positions) == 40:
                # if self.vis:
                #     plot_bev = np.array(self.bev_map_positions)
                #     plot_img = plot(plot_bev[:, 0], plot_bev[:, 1])
                #     cv2.imshow(str(self.id), plot_img)
                move_distance = np.linalg.norm(
                    np.array(self.bev_map_positions[0]) - np.array(self.bev_map_positions[-1]))
                if move_distance / 100 >= self.walking_distance:
                    self.walking_clip = self.frames.copy()
                self.bev_map_positions.pop(0)
                self.frames.pop(0)

            if not self.requested_flag and self.walking_clip is not None:
                Thread(target=self.waiting_rec_api, args=[]).start()
        return self.name

    def waiting_rec_api(self):
        try:
            self.requested_flag = True
            if self.walking_clip is not None:
                if self.save_clip:
                    os.makedirs(f'samples/{self.id}/', exist_ok=True)
                    for idx, f in enumerate(self.walking_clip):
                        cv2.imwrite(f'samples/{self.id}/{idx}.jpg', f)
                info = sent_request_gait(self.walking_clip, track_id=self.id)
                if info['checked']:
                    self.id_code = info['code']
                    self.name = info['fullname']
                    print(self.id, info)
            self.requested_flag = False
        except:
            pass


class Crowd(object):
    def __init__(self, bev_matrix):
        self.bev_mat = bev_matrix
        self.people = {}  # contain {id: person}
        self.ids = []
        self.delete_person_delay = 0.2  # tracking will cancel in 3s

    def update(self, frame, online_tlwhs, online_ids):
        result = {}
        for body, track_id in zip(online_tlwhs, online_ids):
            x1, y1, w, h = body
            body = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            if track_id not in self.people.keys():
                self.people.update({track_id: Person(track_id)})
            name_new = self.people[track_id].update(frame, body, self.bev_mat)
            result.update({track_id: {'box': body, 'name': name_new if name_new is not None else 'none'}})

        return result

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
