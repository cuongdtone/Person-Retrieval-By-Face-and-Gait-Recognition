# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022
import os

from models.human_segment import UnetIR
import cv2
import matplotlib.pyplot as plt
import torch
from models.autoencoder import AE
from models.gait_fc import GaitFCV2
from torchvision import transforms
from PIL import Image
import numpy as np


class GaitEncoding:
    def __init__(self):

        self.seg = UnetIR()
        self.device = torch.device('cpu')
        self.ae = AE()
        self.ae.load_state_dict(torch.load('src/ae_epoch_19_loss_0.04.pt'))
        self.ae.eval()

        transform_list = [transforms.Grayscale(1),
                          transforms.Resize(112),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)

        self.model = GaitFCV2()
        self.model.load_state_dict(torch.load('src/gait_fcv2_epoch_27.pt'))
        self.model.eval()

    def __call__(self, clip, seg=False, show=False):
        # print(len(clip), clip[0].shape)
        ae_feats = []
        gei = np.zeros((112, 112))
        for frame in clip:
            if seg:
                mask = self.seg(frame)
                mask = self.seg.pad(mask)
                mask = self.seg.crop(mask.astype('uint8'))
                mask = cv2.resize(mask, (112, 112))
                mask = mask.astype('uint8')
            else:
                mask = crop(frame)
            mask = Image.fromarray(mask)
            mask = self.transform(mask)
            if show:
                plt.imshow(mask[0])
                plt.show()
            with torch.no_grad():
                code = self.ae.encoder(mask.unsqueeze(0))
            code = code.view(-1).numpy().tolist()
            ae_feats.append(code)
            gei += mask[0].numpy()
        ae_feats = np.array(ae_feats)
        features = self.softmax(ae_feats)
        gei = np.expand_dims(gei, axis=0)/40

        a_v = torch.from_numpy(features).float()
        a_gei = torch.from_numpy(gei).float()

        # print(a_gei.max(), a_gei.min())
        # print(a_v)
        a_v = a_v.unsqueeze(0).to(self.device)
        a_gei = a_gei.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(a_v, a_gei)
        return feat.numpy()[0]

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def compare_tree(feat, user_data, tree):
        # print(feat.shape)
        face_id = tree.get_nns_by_vector(feat, 1)
        feat_db = tree.get_item_vector(face_id[0])
        sim = np.dot(feat, feat_db) / (np.linalg.norm(feat) * np.linalg.norm(feat_db))
        return {'code': user_data[face_id, 0][0], 'fullname': user_data[face_id, 1][0], 'sim': sim}


def crop(mask):
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



