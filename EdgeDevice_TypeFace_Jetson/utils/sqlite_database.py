import sqlite3
from pathlib import Path

QUERY_INSERT_EMP = "INSERT INTO employee ( code, fullname, name, sex, branch, vocative) VALUES (?, ?, ?, ?, ?, ?);"
QUERY_UPDATE_EMP = "UPDATE employee SET fullname = ?, sex = ?, active = ? WHERE code = ?;"
QUERY_UPDATE_STATUS_EMP = "UPDATE employee SET status = ? WHERE code = ?;"
QUERY_INSERT_TKP = """INSERT INTO timekeepings ( code, fullname, image) VALUES (?, ?, ?);"""
QUERY_UPDATE_FEATURE_EMP = "UPDATE employee SET embed = ? WHERE code = ?;"


def connect_database():
    conn = sqlite3.connect('src/TimeKeepingDB.db')
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
        print(ex, '--------------------')
        return False


def get_employee_code(db=None):
    if db == None:
        db = connect_database()
    cur = db.cursor()
    result = cur.execute('SELECT MAX(code) from employee;')
    result_data = result.fetchone()
    cur.close()
    if result == None or len(result_data) > 0:
        return str(int(result_data[0]) + 1)
    else:
        return "10001"


def insert_employee(db, code: str, fullname: str, sex: int, branch: str, isadmin=False):
    if sex == 0:
        vocative = "Anh|Bạn"
    elif sex == 1:
        vocative = "Chị|Bạn"
    else:
        vocative = "Bạn"

    name = str(fullname).split(" ")[-1]

    return execute(db, QUERY_INSERT_EMP, (code, fullname, name, sex, branch, vocative, isadmin))


def update_employee(db, code: str, fullname: str, sex: int, updated_user: str, isadmin=False, status=0, active=False):
    if sex == 0:
        vocative = "Anh|Bạn"
    elif sex == 1:
        vocative = "Chị|Bạn"
    else:
        vocative = "Bạn"

    return execute(db, QUERY_UPDATE_EMP, (fullname, sex, vocative, updated_user, isadmin, status, active, code))


def update_face_feature_employee(db, code: str, feat: str):
    return execute(db, QUERY_UPDATE_FEATURE_EMP, (feat, code))


def update_status_employee(db, code: str, status=1):
    return execute(db, QUERY_UPDATE_STATUS_EMP, (status, code))


def get_all_employee(db):
    try:
        query = "SELECT code, fullname, embed  FROM employee;"
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
        c.execute('DELETE from  timekeepings;')
        db.commit()
    except Exception as e:
        print(e)
        pass


def insert_timekeeping(db, code: str, fullname: str, device_name: str, source='FI', image=''):
    return execute(db, QUERY_INSERT_TKP, (code, fullname, image))
