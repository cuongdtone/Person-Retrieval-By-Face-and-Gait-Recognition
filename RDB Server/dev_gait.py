# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022
import os

import cv2
import torch
from models.autoencoder import AE
from models.gait_fc import GaitFCV2
from torchvision import transforms
from PIL import Image
import numpy as np
from models.gait_recognizer import *


def calc_euclidean(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    return torch.dot(feat1, feat2) / (torch.linalg.norm(feat1) * torch.linalg.norm(feat2))


if __name__ == '__main__':
    source = r'sample\19'
    frames = os.listdir(source)
    clip = []
    for f in frames[:40]:
        image = cv2.imread(os.path.join(source, f), 0)
        mask = crop(image)
        clip.append(image)

    model = GaitEncoding()
    feat1 = model(clip)
    # print(feat)
    # print(feat.shape)

    source = r'sample\10'
    frames = os.listdir(source)
    clip = []
    for f in frames[:40]:
        image = cv2.imread(os.path.join(source, f), 0)
        mask = crop(image)
        clip.append(image)

    feat2 = model(clip)
    # print(feat)
    # print(feat.shape)

    print(calc_euclidean(feat1, feat2))
