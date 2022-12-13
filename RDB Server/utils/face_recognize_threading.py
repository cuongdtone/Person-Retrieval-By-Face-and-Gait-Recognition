# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022
import time

from models.face_recognizer import ArcFaceONNX
from threading import Thread


class FaceRecManager:
    def __init__(self, num_threads=2, data=None, face_tree=None):
        self.num_threads = num_threads
        self.data, self.face_tree = data, face_tree
        self.face_recs = []
        self.queues = []
        for i in range(num_threads):
            self.face_recs.append(ArcFaceONNX())
            self.queues.append([])

        self.outputs = {}
        self.run()

    def get(self, device, track_id, image):
        data = {'device_id': device, 'track_id': track_id, 'img': image}
        min_len = len(self.queues[0])
        thread_ap = 0
        for idx, q in enumerate(self.queues):
            if len(q) < min_len:
                min_len = len(q)
                thread_ap = idx
        if f'{device}-{track_id}' in self.outputs.keys():
            del self.outputs[f'{device}-{track_id}']
        self.queues[thread_ap].append(data)
        while f'{device}-{track_id}' not in self.outputs.keys():
            time.sleep(0.001)
        ouput = self.outputs[f'{device}-{track_id}']
        return ouput

    def run(self):
        for i in range(self.num_threads):
            Thread(target=self.rec, args=[i]).start()

    def rec(self, idx):
        face_model = self.face_recs[idx]
        queue: list = self.queues[idx]
        while True:
            if len(queue) > 0:
                data = queue.pop(0)
                device_id = data['device_id']
                track_id = data['track_id']
                img = data['img']
                # st = time.time()
                feat = face_model(img)
                info = face_model.face_compare_tree(feat, self.data, self.face_tree)
                # print('Model delay: ', time.time() - st)
                self.outputs.update({f'{device_id}-{track_id}': info})
            else:
                time.sleep(0.001)

