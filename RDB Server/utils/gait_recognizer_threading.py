# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022

import time

from models.gait_recognizer import GaitEncoding
from threading import Thread


class GaitRecManager:
    def __init__(self, num_threads=2, data=None, gait_tree=None):
        self.num_threads = num_threads
        self.data, self.gait_tree = data, gait_tree
        self.gait_recs = []
        self.queues = []
        for i in range(num_threads):
            self.gait_recs.append(GaitEncoding())
            self.queues.append([])

        self.outputs = {}
        self.run()

    def get(self, device, track_id, clip, data, gait_tree):
        self.data, self.gait_tree = data, gait_tree
        data = {'device_id': device, 'track_id': track_id, 'clip': clip}
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
            print('Thread start')
            Thread(target=self.rec, args=[i]).start()

    def rec(self, idx):
        model = self.gait_recs[idx]
        queue: list = self.queues[idx]
        while True:
            if len(queue) > 0:
                data = queue.pop(0)
                device_id = data['device_id']
                track_id = data['track_id']
                clip = data['clip']
                # st = time.time()
                feat = model(clip, seg=True)
                info = model.compare_tree(feat, self.data, self.gait_tree)
                # print('Model delay: ', time.time() - st)
                self.outputs.update({f'{device_id}-{track_id}': info})
            else:
                time.sleep(0.001)

