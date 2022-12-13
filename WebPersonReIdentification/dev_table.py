# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 11/29/2022


from flask import Flask, render_template
from datetime import date
from utils.database import *
from calendar import monthrange

month = '11'

people = get_all_people()
timekeep_data = get_timekeep(now_month=month)
# print(date.today().year)
days = monthrange(date.today().year, int(month))
heads = ['ID', "Name", "Position", "Branch"]
show = []

for p in people:

    code = p['code']
    name = p['fullname']
    position = p['position']
    branch = p['branch']
    line1 = [code, name, position, branch]
    # print(timekeep_data)
    timekeep = timekeep_data[code]
    day_temp_1 = ['x'] * (days[1]-days[0]+1)
    day_temp_2 = ['x'] * (days[1] - days[0]+1)

    for d in timekeep:
        time_start = datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S')
        time_end = datetime.datetime.strptime(d[1], '%Y-%m-%d %H:%M:%S')
        # print(time_start.day, day_temp_1)
        print(time_end)
        day_temp_1[time_start.day-1] = time_start.strftime('%H:%M')
        day_temp_2[time_end.day-1] = time_end.strftime('%H:%M')

    line1 = line1 + day_temp_1
    line2 = ['']*4 + day_temp_2
    show.append(line1)
    show.append(line2)
    print(line1)
    print(line2)
    print('''''')
