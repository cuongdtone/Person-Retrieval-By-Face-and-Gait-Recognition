# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022

import cv2
import numpy as np
import onnxruntime


class AutoEncoder:
    def __init__(self):
        self.session = onnxruntime.InferenceSession('src/ae_onnx.onnx')
        self.img_size = 112

    def preprocessing(self, image):
        blob = image.astype('float32') / 127.5 - 1.0
        blob = np.expand_dims(blob, axis=0)
        blob = np.expand_dims(blob, axis=0)
        return blob

    def __call__(self, image):
        image = self.preprocessing(image)
        inp = {'input':  image}
        out = self.session.run(None, inp)[0]
        return out
