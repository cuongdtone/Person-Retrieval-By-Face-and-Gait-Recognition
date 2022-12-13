""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

import cv2
import numpy as np

import onnxruntime
from numpy.linalg import norm as l2norm
from models.face_aligner import norm_crop


def face_compare_tree(feat, user_data, tree):
    face_id = tree.get_nns_by_vector(feat, 1)
    feat_db = tree.get_item_vector(face_id[0])
    sim = np.dot(feat, feat_db) / (np.linalg.norm(feat) * np.linalg.norm(feat_db))
    return {'code': user_data[face_id, 0][0], 'fullname': user_data[face_id, 1][0], 'Sim': sim}


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'

        input_mean = 127.5
        input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        # print('input mean and std:', self.input_mean, self.input_std)
        self.session = onnxruntime.InferenceSession(self.model_file, provider_options=['TensorrtExecutionProvider',
                                                                                       'CUDAExecutionProvider',
                                                                                       'CPUExecutionProvider'])
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def get(self, aimg):
        # aimg = norm_crop(img, landmark=kps)
        embedding = self.get_feat(aimg).flatten()
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
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def face_encoding(self, image, kps):
        aimg = norm_crop(image, landmark=kps)
        feet = self.get(aimg)
        return feet

    def face_compare(self, feet, employees_data, threshold=0.6):
        max_sim = -1
        info = {'fullname': 'uknown', 'Sim': max_sim, 'code': None}
        if employees_data is not None:
            for data in employees_data:
                feet_compare = data[3]
                sim = self.compute_sim(feet, feet_compare)
                if sim > threshold and sim > max_sim:
                    max_sim = sim
                    info['fullname'] = data[1]
                    info['Sim'] = max_sim
                    info['code'] = data[0]
        return info


class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        # for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'
