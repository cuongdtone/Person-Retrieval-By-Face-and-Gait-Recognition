""" Created by MrBBS """
# 2/10/2022
# -*-encoding:utf-8-*-

import numpy as np
import onnxruntime
import cv2


class MaskDetection:
    def __init__(self, weight='model_mask.onnx'):
        self.sess = onnxruntime.InferenceSession(weight,
                                                 provider_options=['TensorrtExecutionProvider',
                                                                   'CUDAExecutionProvider',
                                                                   'CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def __call__(self, image_face):
        """
        iamge_face: norm crop face
        Kiểm tra khuôn mặt có đeo khẩu trang hay không
        :param image_face: hình ảnh ( np.array ) face
        :return: True / False ( Không đeo / Đeo )
        """
        h, w = image_face.shape[:2]
        image_face = image_face[h // 6:h, w // 5:w - w // 5]
        image_face = cv2.resize(cv2.cvtColor(image_face, cv2.COLOR_BGR2GRAY), (16, 16))
        image_face = np.expand_dims(np.expand_dims(image_face, axis=2), axis=0).astype(np.float32)
        pred = self.sess.run([self.output_name], {self.input_name: image_face})[0]
        return np.argmax(pred) == 1