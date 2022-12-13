# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022

import cv2
import numpy as np
import onnxruntime
from models.human_segment import UnetIR
from models.ae_gait import AutoEncoder


class GaitCal:
    def __init__(self):
        self.session = onnxruntime.InferenceSession('src/gait_cal.onnx')
        self.img_size = 112

        self.seg = UnetIR()
        self.ae = AutoEncoder()

    def __call__(self, clip):
        ge, al = self.part1_with_mask(clip)
        al = np.expand_dims(al, axis=0)
        ge = np.expand_dims(ge, axis=0)
        # print(al.shape, ge.shape)
        embedding = self.part2(al, ge)
        return embedding

    def part2(self, al, ge):
        # print(al.shape)
        inp = {'al': al, 'ge': ge}
        out = self.session.run(None, inp)[0]
        return out

    def part1(self, frames):
        ae_feats = []
        gei = np.zeros((112, 112))
        for image in frames:
            o = self.seg(image)
            mask = self.seg.pad(o)
            mask = self.crop(mask.astype('uint8'))
            mask = cv2.resize(mask, (112, 112))
            gei += mask / 255
            ae_feat = self.ae(mask)
            ae_feats.append(ae_feat)
        gei = np.expand_dims(gei / 40, axis=0)
        ae_feats = np.array(ae_feats)
        return gei.astype('float32'), ae_feats.astype('float32')

    def part1_with_mask(self, frames):
        ae_feats = []
        gei = np.zeros((112, 112), dtype='float32')
        for frame in frames:
            frame = self.crop(frame.astype('uint8'))
            mask = cv2.resize(frame, (112, 112))
            code = self.ae(mask)
            ae_feats.append(code)
            gei += mask/255
        ae_feats = np.array(ae_feats)
        gei = np.expand_dims(gei / 40, axis=0)
        return gei.astype('float32'), ae_feats.astype('float32')

    @staticmethod
    def crop(mask):
        mask = mask[:, :, 0]
        x, y, w, h = cv2.boundingRect(mask)
        crop = mask[y:y + h, x:x + w]
        if h > w:
            bg = np.zeros((h, h))
            st_x = int(h / 2 - w / 2)
            ed_x = int(h / 2 + w / 2)
            bg[:, st_x:ed_x] = crop
        else:
            bg = np.zeros((w, w))
            st_y = int(w / 2 - h / 2)
            ed_y = int(w / 2 + h / 2)
            bg[st_y:ed_y, :] = crop
        return bg
