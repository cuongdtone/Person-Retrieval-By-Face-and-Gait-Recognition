# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/23/2022

import torch
import torch.nn as nn


class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        # self.embedding = nn.Embedding(max_features, embed_size)
        # self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # self.embedding.weight.requires_grad = False
        embed_size = 324
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 128)

    def forward(self, x):
        # h_embedding = self.embedding(x)
        # h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

        h_lstm, _ = self.lstm(x)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        # print("avg_pool", avg_pool.size())
        # print("max_pool", max_pool.size())
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out



if __name__ == '__main__':
    from config import *
    import torch
    import time
    net = BiLSTM()
    # print(net)
    net.eval()

    x = torch.rand((1, 40, 324))
    st = time.time()
    y = net(x)
    print(time.time() - st)
    print(y.shape)



