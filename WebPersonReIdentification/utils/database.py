# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 11/30/2022

import sqlite3
from pathlib import Path
from cfg import cfg
import datetime


def connect_database():
    conn = sqlite3.connect(cfg.sqdb)
    return conn


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def execute(db, query, values=None):
    try:
        if db == None:
            db = connect_database()
        cur = db.cursor()
        cur.execute(query, values)
        db.commit()

        cur.close()
        return True
    except Exception as ex:
        db.rollback()
        return False


def get_all_people(db=None):
    try:
        query = "SELECT * FROM people;"
        if db is None:
            db = connect_database()
        cur = db.cursor()
        result = cur.execute(query)
        result_data = result.fetchall()
        employee = []
        for row in result_data:
            employee.append(dict_factory(cur, row))
        cur.close()
        return list(employee)
    except Exception as ex:
        print(ex)
        return None


def clear_tab(db=None):
    try:
        if db is None:
            db = connect_database()
        c = db.cursor()
        c.execute('DELETE from tracing;')
        db.commit()
    except Exception as e:
        print(e)
        pass


def update_gait(code, gaits):
    QUERY_UPDATE_STATUS_EMP = "UPDATE people SET gait_embed = ? WHERE code = ?;"
    return execute(None, QUERY_UPDATE_STATUS_EMP, (gaits, code))


def get_timekeep(db=None, now_month='11', year='2022'):
    try:
        if db is None:
            db = connect_database()
        cur = db.cursor()

        # now_month = '11' #str(datetime.datetime.now().month)

        query_name = "SELECT code, fullname FROM people;"
        query = "SELECT * FROM tracing WHERE strftime('%m', checkin) = ? AND strftime('%Y', checkin) = ? AND code = ?;"

        raw = cur.execute(query_name)
        raw = raw.fetchall()
        people_infor = [i[:] for i in raw]

        timekeep_data = {}

        for id_code, fullname in people_infor:
            # print(fullname)
            raw_ = cur.execute(query, [now_month, year, id_code])
            raw_ = raw_.fetchall()
            timekeep = []
            for d in range(1, 31):
                # print('    Date ', d)
                start = None
                end = None

                length_raw = len(raw_)
                for i in range(length_raw):
                    time_ = raw_[i][2]
                    time_ = datetime.datetime.strptime(time_, '%Y-%m-%d %H:%M:%S')
                    # print(int(time_.day) == int(d), id_code == raw_[i][0])
                    if id_code == raw_[i][0] and int(time_.day) == int(d):
                        # print('    ', raw_[i])
                        if start is None:
                            start = raw_[i][2]
                        end = raw_[i][2]
                if start is not None and end is not None:
                    timekeep.append([start, end])
            timekeep_data.update({id_code: timekeep})
        cur.close()
        return timekeep_data
    except Exception as ex:
        print(ex)
        return None


if __name__ == '__main__':
    raw_data = get_timekeep()
    print(raw_data)
