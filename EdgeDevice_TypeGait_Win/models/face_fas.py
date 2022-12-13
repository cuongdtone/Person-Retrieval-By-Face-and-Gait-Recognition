# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/13/2022

import onnxruntime
import cv2
import numpy as np
from openvino.runtime import Core


class FAS:
    def __init__(self):
        model_file = 'src/ir_model/face_fas/ae_fas.xml'

        core = Core()
        model_ir = core.read_model(model=model_file)
        self.compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")
        self.output_layer_ir = self.compiled_model_ir.output(0)

        self.image_shape = [224, 224]

    def __call__(self, image):
        inp = self.preprocess(image)
        pred = self.compiled_model_ir([inp])[self.output_layer_ir]
        prob = self.softmax(pred)[0]
        print(prob)
        if prob[0] < 0.01:
            return True
        return False

    def preprocess(self, image):
        img_resized = cv2.resize(image, (self.image_shape[0], self.image_shape[1]))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype('float32')/255.0
        img_resized = img_resized.transpose(2, 0, 1)
        img_resized = np.expand_dims(img_resized, axis=0)
        return img_resized

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


if __name__ == '__main__':
    fas = FAS()
    image = cv2.imread('cc.png')
    pred = fas(image)
    print(pred)
