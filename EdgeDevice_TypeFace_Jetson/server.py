""" Created by MrBBS """
# 10/17/2022
# -*-encoding:utf-8-*-

import json
import sqlite3
import time

import ciso8601
from flask import Flask, request, jsonify
from pathlib import Path

path_voice = Path('media')


def connect_to_db():
    conn = sqlite3.connect('src/TimeKeepingDB.db')
    # conn = sqlite3.connect('TimeKeepingDB.db')
    return conn


def insert_employee(idx, name, emb):
    exc = 'INSERT OR REPLACE INTO employee (code, fullname, embed) VALUES (?, ?, ?)'
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute(exc, (idx, name, str(emb)))
    conn.commit()
    cur.close()
    return True


def get_timekeep(date_btw):
    timekeeps = []

    def _get(d):
        try:
            conn = connect_to_db()
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM timekeepings WHERE working_date BETWEEN '{d['from_date']}' AND '{d['to_date']}'")
            rows = cur.fetchall()
            for i in rows:
                working_date = time.mktime(ciso8601.parse_datetime(i["working_date"]).timetuple())
                checkin = time.mktime(ciso8601.parse_datetime(i['checkin']).timetuple())
                timekeeps.append({"code": i["code"], "fullname": i["fullname"], "working_date": working_date,
                                  "checkin": checkin})

        except Exception as e:
            return e

    if isinstance(date_btw, list):
        for d in date_btw:
            _get(d)
    else:
        _get(date_btw)
    return timekeeps


def get_timekeep_by_id(data):
    timekeeps = []

    def _get(d):
        try:
            conn = connect_to_db()
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM timekeepings WHERE code = '{d['code']}' AND working_date BETWEEN '{d['from_date']}' AND '{d['to_date']}'")
            rows = cur.fetchall()
            for i in rows:
                working_date = time.mktime(ciso8601.parse_datetime(i["working_date"]).timetuple())
                checkin = time.mktime(ciso8601.parse_datetime(i['checkin']).timetuple())
                timekeeps.append({"code": i["code"], "fullname": i["fullname"], "working_date": working_date,
                                  "checkin": checkin, "image": str(i["image"])})

        except:
            pass

    if isinstance(data, list):
        for d in data:
            _get(d)
    else:
        _get(data)

    return timekeeps


def clean_timekeep(date):
    try:
        db = connect_to_db()
        c = db.cursor()
        c.execute(f"DELETE FROM timekeepings WHERE working_date <= '{date}'")
        db.commit()
    except Exception as e:
        return False
    return True


def delete_employee(data):
    try:
        conn = connect_to_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if isinstance(data, list):
            code = data[0]['code']
        else:
            code = data['code']
        cur.execute(f"DELETE FROM employee WHERE code = '{code}'")
        conn.commit()

    except:
        return False
    return True


app = Flask(__name__)


@app.route('/api/get_config', methods=['GET'])
def api_get_config():
    with open('src/format_audio.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
    return jsonify(config)


@app.route('/api/timekeep', methods=['POST'])
def api_get_timekeep():
    data = request.get_json()
    data = json.loads(data)
    check = data
    if isinstance(data, list):
        check = data[0]
    if 'code' in check.keys():
        return jsonify(get_timekeep_by_id(data))
    return jsonify(get_timekeep(data))


@app.route('/api/employee/add', methods=['POST'])
def api_add_employee():
    body = request.form
    files = request.files.getlist('files')
    if len(files) > 0:
        with open('src/format_audio.json', 'r', encoding='utf-8') as f:
            config = json.loads(f.read())
        check = {}
        for k in config.keys():
            check.setdefault(k, 1)
        code = body['code']
        for file in files:
            num = check.get(file.filename, 0)
            if num > 0:
                folder = path_voice.joinpath(file.filename)
                folder.mkdir(parents=True, exist_ok=True)
                file.save(folder.joinpath(f"{code}_{num}.wav").as_posix())
                check[file.filename] += 1
    return jsonify(insert_employee(body['code'], body['fullname'], body['emb']))


@app.route('/api/employee/delete', methods=['POST'])
def api_delete_employee():
    data = request.get_json()
    data = json.loads(data)
    return jsonify(delete_employee(data))


@app.route('/api/clean', methods=['POST'])
def api_clean_timekeeps():
    data = request.get_json()
    data = json.loads(data)
    return jsonify(clean_timekeep(data['date']))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
    # app.run()
