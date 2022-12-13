# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 11/29/2022

from flask import Flask, render_template, request
from datetime import date
from utils.database import *
from utils.api import *
from calendar import monthrange

app = Flask(__name__)


@app.route('/login/', methods=['GET', 'POST'])
def login():
    return render_template('login2.html')


@app.route('/login/submit', methods=['GET', 'POST'])
def login_submit():
    # user_name = request.args.get('submit') #todo: add login feature
    return render_template('dashboard.html')


@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    if request.method == 'GET':
        return render_template('add_employee.html')
    else:
        data_form = dict(request.form)
        data = {'fullname': data_form['firstName'] + ' ' + data_form['lastName'], 'code': data_form['ID'],
                'position': data_form['position'], 'birthday': data_form['birthday'],
                'branch': data_form['branch'],
                'vocative': 'Anh' if data_form['vocative'] == 'male' else 'Chị'}

        add_person_to_server(data)
        return render_template('add_employee.html')


@app.route('/employees')
def employees():
    raw_data = get_all_em_from_server()
    data = []
    for idx, p in enumerate(raw_data):
        p['face_embed'] = 'Yes' if p['face_embed'] is not None else 'No'
        p['gait_embed'] = 'Yes' if p['gait_embed'] is not None else 'No'

        p = (p['code'], p['fullname'], p['vocative'], p['birthday'],
             p['create_time'], p['branch'], p['position'], p['face_embed'], p['gait_embed'])
        data.append(p)
    return render_template('employees.html', data=tuple(data))


@app.route('/timekeep', methods=['GET', 'POST'])
def timekeep():

    if request.method == 'POST':
        try:
            month, year = request.form['month'].split('/')
            people = get_all_em_from_server()
            timekeep_data = get_timekeep_from_server(month=month, year=year)
            # print(date.today().year)
            days = (1, monthrange(date.today().year, int(month))[1])
            heads = ['ID', "Name", "Position", "Branch"]
            show = []
            show_days = range(days[0], days[1]+1)
            for p in people:

                code = p['code']
                name = p['fullname']
                position = p['position']
                branch = p['branch']
                line1 = [code, name, position, branch]
                # print(timekeep_data)
                timekeep = timekeep_data[code]
                day_temp_1 = ['x'] * (days[1] - days[0] + 1)
                day_temp_2 = ['x'] * (days[1] - days[0] + 1)

                for d in timekeep:
                    time_start = datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S')
                    time_end = datetime.datetime.strptime(d[1], '%Y-%m-%d %H:%M:%S')
                    # print(time_start.day, day_temp_1)
                    day_temp_1[time_start.day - 1] = time_start.strftime('%H:%M')
                    day_temp_2[time_end.day - 1] = time_end.strftime('%H:%M')

                line1 = line1 + day_temp_1
                line2 = [''] * 4 + day_temp_2
                show.append(line1)
                show.append(line2)

            return render_template('timekeep.html', data=tuple(show), head=tuple(list(heads) + list(show_days)))
        except:
            return render_template('timekeep.html')
    else:
        return render_template('timekeep.html')


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)
