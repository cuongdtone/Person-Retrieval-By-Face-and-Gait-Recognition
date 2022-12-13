""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

import threading
import multiprocessing as mp
import cv2
import os
import os.path as osp
from pathlib import Path
import pickle
import time
import sys

import numpy as np


def get_object(name):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


class BuffelessCamera(threading.Thread):
    def __init__(self, rtsp_url, name='camera-buffer-cleaner-thread'):
        self.ret = False
        self.rtsp = rtsp_url
        self.camera = cv2.VideoCapture(rtsp_url)
        self.fps = 25
        self.last_frame = None
        super(BuffelessCamera, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            self.ret, self.last_frame = self.camera.read()
            if not self.ret:
                self.ret = False
                self.last_frame = np.zeros((640, 640, 3), dtype='uint8')
                self.reconnect()

    def reconnect(self):
        # print('Reconnecting !')
        while True:
            self.camera.release()
            del self.camera
            time.sleep(0.1)
            self.camera = cv2.VideoCapture(self.rtsp)
            ret, last_frame = self.camera.read()
            if ret:
                break
        # print("Reconnect sucessfully !")

    def read(self):
        return self.ret, self.last_frame

    def isOpened(self):
        return True

    def get_resolution(self):
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return (height, width)


class CustomThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class Watcher(threading.Thread):
    running = True
    refresh_delay_secs = 1

    def __init__(self, watch_file, call_func_on_change=None, *args, **kwargs):
        self._cached_stamp = 0
        self.filename = watch_file
        self.call_func_on_change = call_func_on_change
        self.args = args
        self.kwargs = kwargs
        super(Watcher, self).__init__()
        self.start()

    def look(self):
        stamp = os.stat(self.filename).st_mtime
        if stamp != self._cached_stamp:
            self._cached_stamp = stamp
            # File has changed, so do something...
            if self.call_func_on_change is not None:
                self.call_func_on_change(*self.args, **self.kwargs)

    def run(self):
        while self.running:
            try:
                # Look for changes
                time.sleep(self.refresh_delay_secs)
                self.look()
            except KeyboardInterrupt:
                print('\nDone')
                break
            except FileNotFoundError:
                # Action on file not found
                pass
            except:
                print('Unhandled error: %s' % sys.exc_info()[0])
