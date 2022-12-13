# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 12/13/2022

from cfg import cfg
import requests


def get_all_em_from_server():
    url = f'{cfg["server"]}/api/get_people_table'
    response = requests.get(url)
    return response.json()['data']


def get_timekeep_from_server(month, year):
    url = f'{cfg["server"]}/api/get_timekeep'
    data = {'month': month, "year": year}
    response = requests.get(url, json=data)
    return response.json()['data']


def add_person_to_server(data):
    url = f'{cfg["server"]}/api/add_person_to_db'
    response = requests.post(url, json=data)
    return response.json()
