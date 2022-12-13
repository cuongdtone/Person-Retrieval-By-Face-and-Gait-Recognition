# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 11/1/2022

from models.human_segment import UnetIR
import cv2
import torch
from models.autoencoder import AE
from models.gait_fc import GaitFCV2
from torchvision import transforms
from PIL import Image
import numpy as np
from models.gait_recognizer import *
import glob
from utils.sqlite_db import update_gait


def calc_euclidean(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    return torch.dot(feat1, feat2) / (torch.linalg.norm(feat1) * torch.linalg.norm(feat2))


model = GaitEncoding()


if __name__ == '__main__':
    vs = ['yen_street_2', 'yen_street_3', 'yen_street_4', 'yen_street_5', 'yen_street_6']
    gaits = []
    for v in vs:
        images = glob.glob(fr'E:\Person Retrieval\Person Detect\samples/{v}/*')
        clip = []
        for idx, i_path in enumerate(images):
            image = cv2.imread(i_path)
            clip.append(image)
        embed1 = model(clip, seg=True, show=True)
        gaits.append(embed1.tolist())

    update_gait('15', str(gaits))

    # print(embed1)

    # v = 'cuong_3'
    # images = glob.glob(fr'../Person Detect/samples/{v}/*')
    # clip = []
    # for idx, i_path in enumerate(images):
    #     image = cv2.imread(i_path)
    #     clip.append(image)
    # embed2 = model(clip, seg=True)
    # # print(embed2)
    #
    # print(calc_euclidean(embed1, embed2))


