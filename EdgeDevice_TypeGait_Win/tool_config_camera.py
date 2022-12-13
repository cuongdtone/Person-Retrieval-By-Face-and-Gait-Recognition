# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 11/25/2022

import time

import cv2
import numpy as np
import requests


class WalkingSize:
    def __init__(self, source=r'samples/test_street.mp4', M_path='src/M_camera3.npy'):
        self.M_path = M_path
        self.meter2pixel = 100
        cv2.namedWindow('cc', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('cc', 1280, 720)

        # cv2.namedWindow('cc2', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('cc2', 800, 500)

        cv2.setMouseCallback('cc', self.draw_circle)
        self.cap = cv2.VideoCapture(source)
        _, self.frame = self.cap.read()

        self.poly_area = []
        self.area_select_flag = True

    def run(self):
        while True:
            cv2.imshow('cc', self.frame)
            if cv2.waitKey(2) & 0xFF == 27:
                break

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # check if mouse event is click
            if self.area_select_flag:
                self.poly_area.append([x, y])
                cv2.circle(self.frame, (x, y), 5, (255, 0, 0), -1)  # draw filled circle with 100px radius
                if len(self.poly_area) == 4:
                    cv2.polylines(self.frame, [np.array(self.poly_area)], isClosed=True, color=(0, 255, 0), thickness=2)
                    self.area_select_flag = False
                    time.sleep(1)
                    self.w = float(input('Width (m): '))
                    self.h = float(input('Height (m): '))
                    self.get_m()
                cv2.polylines(self.frame, [np.array(self.poly_area)], isClosed=False, color=(0, 255, 0), thickness=2)
            else:
                cv2.circle(self.frame, (x, y), 5, (255, 0, 0), -1)  # draw filled circle with 100px radius

    def get_m(self):
        w = int(self.w * self.meter2pixel)
        h = int((w/self.w) * self.h)
        print(h)
        dst, M = self.four_point_transform(self.frame, self.poly_area, w, h)
        np.save(self.M_path, M)
        # print(self.warp_point(0, 0, M))
        cv2.imshow('cc2', dst)
        cv2.waitKey()

    @staticmethod
    def four_point_transform(img, polygon, w, h):
        dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype="float32")
        pts = np.array(polygon, dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (w, h))
        return warped, M

    @staticmethod
    def warp_point(x: int, y: int, M):
        d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
        return (
            int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
            int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
        )


if __name__ == '__main__':
    WalkingSize().run()
    # url = f'http://192.168.1.160:5000/api/cap'
    # response = requests.post(url)
    # print(response)


