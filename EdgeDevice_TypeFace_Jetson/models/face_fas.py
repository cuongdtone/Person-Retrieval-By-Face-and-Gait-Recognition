# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/13/2022

import onnxruntime
import cv2
import numpy as np


class FAS:
    def __init__(self, weight='src/ae_fas.onnx'):
        self.session = onnxruntime.InferenceSession(weight,
                                                    provider_options=['TensorrtExecutionProvider',
                                                                      'CUDAExecutionProvider',
                                                                      'CPUExecutionProvider'])

        self.outputs_name = [i.name for i in self.session.get_outputs()]
        self.image_shape = [224, 224]

    def __call__(self, image):
        # cv2.imshow('fas', image)
        inp = {'input': self.preprocess(image)}
        pred = self.session.run(self.outputs_name, inp)[0]
        prob = self.softmax(pred)[0]
        # print(prob)
        if prob[0] < 0.01:
            return True
        return False

    def preprocess(self, image):
        img_resized = cv2.resize(image, (self.image_shape[0], self.image_shape[1]))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype('float32')/255.0
        img_resized = img_resized.transpose(2, 0, 1)
        img_resized = np.expand_dims(img_resized, axis=0)
        return img_resized

    # def preprocess(self, image):
    #     img_resized = cv2.resize(image, (self.image_shape[0], self.image_shape[1]))
    #     scale = 1 / 255.0
    #     input_blob = cv2.dnn.blobFromImage(
    #         image=img_resized,
    #         scalefactor=scale,
    #         size=img_resized.shape[:2][::-1],  # img target size
    #         swapRB=True,  # BGR -> RGB
    #     )
    #     return input_blob

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
