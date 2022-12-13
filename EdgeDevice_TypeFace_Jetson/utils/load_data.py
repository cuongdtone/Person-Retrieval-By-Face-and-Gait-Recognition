""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

from utils.api import get_all_em_from_server
import os
import numpy as np
from annoy import AnnoyIndex


def load_user_data():
    users_info = get_all_em_from_server()['data']
    tree = AnnoyIndex(512, 'euclidean')
    data = []
    count = 0
    for user in users_info:
        if user["face_embed"] is not None and len(user["face_embed"]) != 0:
            info = [user['code'], user['fullname'], 'name']
            feat = str2list(user['face_embed'])
            tree.add_item(count, feat)
            data.append(info)
            count += 1
    tree.build(100)
    data = np.array(data)
    return data, tree, users_info


def str2list(face_feature):
    str = face_feature.strip(']')[1:]
    return [float(i.strip('\n')) for i in str.split()]
