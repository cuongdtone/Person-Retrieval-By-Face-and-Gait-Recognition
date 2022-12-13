# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 11/15/2022


import glob
import time
from threading import Thread
from utils.api import *

num_request = [1, 4, 8, 16, 32, 64]
delay = []


def calc_api(*args):
    st = time.time()
    sent_request_gait(*args)
    delay.append(time.time() - st)


if __name__ == '__main__':

    images = glob.glob(fr'samples/yen_street_2\*')
    clip = []
    for idx, i_path in enumerate(images):
        image = cv2.imread(i_path)
        # image = resize(image, 400)
        # print(image.shape)
        clip.append(image)

    for n in num_request:
        for _ in range(3):
            threads = []
            for _ in range(n):
                # st = time.time()
                t = Thread(target=calc_api, args=(clip, 1, 1))
                t.start()
                threads.append(t)
                # print(time.time() - st)

            for t in threads:
                t.join()

        print(f'Num request = {n}, time=', sum(delay)/len(delay))

