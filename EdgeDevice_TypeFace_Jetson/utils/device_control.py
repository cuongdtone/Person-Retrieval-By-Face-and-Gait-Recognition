# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/19/2022
import time

from playsound import playsound
from threading import Thread
from queue import Queue


class LinearDevice:
    def __init__(self):
        self.queue = []

    def start(self):
        a = Thread(target=self.run, args=[self.queue])
        a.start()

    @staticmethod
    def run(queue):
        while True:
            time.sleep(0.5)
            try:
                if len(queue) != 0:
                    path = queue.pop(0)
                    playsound(path)
            except Exception as e:
                pass


if __name__ == '__main__':
    device = LinearDevice()
    device.start()
    time.sleep(1)
    device.queue.append(r'D:\DOAN\FaceLognew\media/goodbye\13_1.wav')
    # device.queue.append(r'D:\FaceLog\media\hello\3.wav')
