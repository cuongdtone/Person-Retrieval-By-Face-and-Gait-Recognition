# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/25/2022


import torch
import torch.nn as nn

from models.lstm import BiLSTM
from models.mbf import get_mbf


class GaitFC(nn.Module):
    def __init__(self):
        super().__init__()

        self.gei_encoder = get_mbf(True, 128)
        self.ae_sequence = BiLSTM()
        # self.gei_encoder.train()
        # self.ae_sequence.train()

    def forward(self, ae_feat, gei):
        rnn_feat = self.ae_sequence(ae_feat)
        # print(gei.shape)
        gei_feat = self.gei_encoder(gei)
        feat = torch.concat([rnn_feat, gei_feat], dim=-1)
        return feat

        # print(feat.shape)


class GaitFCV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.gei_encoder = get_mbf(True, 128)
        self.ae_sequence = BiLSTM()
        # self.gei_encoder.train()
        # self.ae_sequence.train()
        self.fc = nn.Linear(256, 128)

    def forward(self, ae_feat, gei):
        rnn_feat = self.ae_sequence(ae_feat)
        # print(gei.shape)
        gei_feat = self.gei_encoder(gei)
        feat = torch.concat([rnn_feat, gei_feat], dim=-1)
        # feat = self.fc(feat)
        return feat

        # print(feat.shape)


if __name__ == '__main__':
    model = GaitFCV2()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # for parameter in model.parameters():
    #     print(parameter)
    x_ae = torch.rand(2, 40, 324)
    x_gei = torch.rand(2, 1, 112, 112)

    y = model(x_ae, x_gei)
    print(y.shape)
