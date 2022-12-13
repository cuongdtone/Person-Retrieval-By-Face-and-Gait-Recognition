# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022

import cv2
import numpy as np
import onnxruntime
from openvino.runtime import Core


class ArcFaceONNX:
    def __init__(self, model_file='src/arcface_ir/w600k_mbf.xml'):

        input_mean = 127.5
        input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std

        core = Core()
        model_ir = core.read_model(model=model_file)
        self.compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")
        self.output_layer_ir = self.compiled_model_ir.output(0)

    def __call__(self, face_img):
        embedding = self.get_feat(face_img).flatten()
        return embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        img = imgs[..., ::-1]
        blob = img.astype('float32') / 127.5 - 1.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        net_outs = self.compiled_model_ir([blob])[self.output_layer_ir]
        return net_outs

    @staticmethod
    def face_compare_tree(feat, user_data, tree):
        face_id = tree.get_nns_by_vector(feat, 1)
        feat_db = tree.get_item_vector(face_id[0])
        sim = np.dot(feat, feat_db) / (np.linalg.norm(feat) * np.linalg.norm(feat_db))
        return {'code': user_data[face_id, 0][0], 'fullname': user_data[face_id, 1][0], 'sim': sim}



