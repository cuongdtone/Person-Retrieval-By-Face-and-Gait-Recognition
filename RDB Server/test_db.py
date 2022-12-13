# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 10/30/2022

from utils.sqlite_db import insert_new_person
from utils.load_data import load_user_data

insert_new_person('code', 'fullname', 'vocative', 'birthday', 'branch', 'position')
