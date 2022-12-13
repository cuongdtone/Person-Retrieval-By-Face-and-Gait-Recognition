# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 12/12/2022

import requests
import yaml


with open("settings.yaml", 'r') as f:
    cfg = yaml.safe_load(f)


def get_all_em_from_server():
    url = f'{cfg["server"]}/api/face_recognize'
    response = requests.get(url)
    return response.json()


def insert_timekeeping_to_server(code: str, fullname: str):
    url = f'{cfg["server"]}/api/insert_face_log'
    data = {'code': code, "fullname": fullname, 'device': cfg['device']}
    response = requests.get(url, json=data)
    return response


insert_timekeeping_to_server('1', '2')

