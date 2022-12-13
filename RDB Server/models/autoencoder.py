# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/24/2022
import torch
from torch import nn
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 4, 3, stride=2, padding=1),  # b, 8, 3, 3
            # nn.BatchNorm2d(4),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 3, stride=2),  # b, 16, 5, 5
            nn.Hardswish(inplace=True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.Hardswish(inplace=True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
    #
    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = self.decoder(x)
    #     return x

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1)
        return x #F.softmax(x)


if __name__ == '__main__':
    model = AE()
    x = torch.rand((1, 1, 112, 112))
    print(x.shape)
    y = model.encoder(x)
    y = y.view(-1)
    y2 = model(x)
    print(y2.shape)
    # print(y)
    print(y.shape)
