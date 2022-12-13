# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022


from utils.sqlite_db import get_all_people
import os
import numpy as np
from annoy import AnnoyIndex


def load_user_data():
    users_info = get_all_people()
    face_tree = AnnoyIndex(512, 'euclidean')
    gait_tree = AnnoyIndex(256, 'euclidean')
    data = []

    count = 0
    for user in users_info:
        info = [user['code'], user['fullname']]
        append_info = 0

        if user["gait_embed"] is not None and len(user["gait_embed"]) != 0:
            gait_feat = eval(user['gait_embed'])
            for f in gait_feat:
                gait_tree.add_item(count, f)
                count += 1
            append_info = len(gait_feat)
        for _ in range(append_info):
            data.append(info)
    face_tree.build(100)
    gait_tree.build(100)
    data = np.array(data)
    return data, face_tree, gait_tree


def str2list(face_feature):
    s = face_feature.strip(']')[1:]
    return [float(i.strip('\n')) for i in s.split()]
