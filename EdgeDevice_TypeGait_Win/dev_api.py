# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022

import glob

from utils.api import *


def resize(image, window_height=300):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    window_width = window_height / aspect_ratio
    image = cv2.resize(image, (int(window_height), int(window_width)))
    return image


if __name__ == '__main__':
    vs = ['yen_street_1', 'yen_street_1', 'yen_street_1']
    v = vs[1]
    # images = glob.glob(fr'E:\Person Retrieval\RDB Server\sample\yen_3\*')
    images = glob.glob(fr'E:\Person Retrieval\Person Detect\samples\yen_street_4/*')
    clip = []
    for idx, i_path in enumerate(images):
        image = cv2.imread(i_path)
        clip.append(image)
    st = time.time()
    info = sent_request_gait(clip, 1)
    print(info)
    print(time.time() - st)
