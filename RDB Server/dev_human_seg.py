# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022

import glob
import os

import cv2
import numpy as np
import pickle
from models.gait_cal import GaitCal
from models.human_segment import UnetIR


seg = UnetIR()

v = 'mask'

os.makedirs(f'sample/{v}', exist_ok=True)
images = glob.glob(fr'E:\Person Retrieval\Person Detect\samples\cuong_1\*')
clip1 = []
for idx, i_path in enumerate(images):
    print(i_path)
    image = cv2.imread(i_path)
    print(image.shape)
    mask = seg(image)
    mask = seg.pad(mask)
    mask = seg.crop(mask.astype('uint8'))
    mask = cv2.resize(mask, (112, 112))
    cv2.imwrite(f'sample/{v}/%3d.jpg' % idx, mask)
