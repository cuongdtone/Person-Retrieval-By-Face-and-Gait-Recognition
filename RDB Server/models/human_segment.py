# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022


import albumentations as albu
import cv2
import numpy as np
from openvino.runtime import Core

from utils.unet_utils import *


class UnetIR:
    def __init__(self):
        core = Core()
        model_ir = core.read_model(model="src/unet_ir/unet_segment.xml")
        self.compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")
        self.output_layer_ir = self.compiled_model_ir.output(0)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
        x = self.normalize(padded_image, self.mean, self.std)
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        res_ir = self.compiled_model_ir([x])[self.output_layer_ir][0][0]
        mask = (res_ir > 0).astype(np.uint8) * 255
        return mask

    @staticmethod
    def pad(mask):
        h, w = mask.shape
        if h > w:
            pad_size = int((h-w)/2)
            out = np.zeros((h, h))
            out[:, pad_size:h-pad_size] = mask
        else:
            pad_size = int((w-h)/2)
            out = np.zeros((w, w))
            out[pad_size:w-pad_size, :] = mask
        return np.expand_dims(out, axis=-1)

    @staticmethod
    def normalize(img, mean, std, max_pixel_value=255.0):
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value

        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

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
