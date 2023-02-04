import json
import pandas as pd
import psycopg2
from datetime import datetime


# db 연결
db_secrets_path = "/opt/ml/input/code/fastapi/app/back/db_secrets.json"
with open(db_secrets_path, "r") as f:
    db_secrets = json.load(f)
db = psycopg2.connect(
    host=db_secrets["DB"]["host"],
    dbname=db_secrets["DB"]["dbname"],
    user=db_secrets["DB"]["user"],
    password=db_secrets["DB"]["password"],
    port=db_secrets["DB"]["port"],
)


def get_info_from_db(exam_info):
    answer = pd.read_sql(
        f'select "QUESTION_PK", "ANSWER" from "ANSWER_TB" Where "TYPE_PK" like \'%{exam_info}%\';',
        db,
    )
    question = pd.read_sql(
        f'select "QUESTION_PK", x, y, w, h, page from "QUESTION_BBOX_TB" Where "TYPE_PK" like \'%{exam_info}%\';',
        db,
    )
    img_shape = pd.read_sql(
        f'select "WIDTH", "HEIGHT" from "PAGE_SHAPE" where "TYPE_PK" like \'%{exam_info}%\';',
        db,
    )

    answer = {q: a for q, a in zip(answer["QUESTION_PK"], answer["ANSWER"])}
    q_bbox = [[dict()] for _ in range(max(question["page"]))]
    for q, x, y, w, h, page in zip(
        question["QUESTION_PK"],
        question["x"],
        question["y"],
        question["w"],
        question["h"],
        question["page"],
    ):
        q_bbox[page - 1][0].update({q: [x, y, w, h]})
    img_shape = [img_shape["WIDTH"][0], img_shape["HEIGHT"][0]]

    # db.close() # 매번 close?
    return answer, q_bbox, img_shape


def insert_log(log_pred, exam_info, log_name):
    time = datetime.now()
    db.cursor().execute(
        f'insert into "LOG_UUID" ("INFER_DATE", "UUID") \
                                values (\'{time}\', \'{log_name}\');'
    )
    for p in log_pred:
        x, y, w, h, confidence, question_num, pred = p
        db.cursor().execute(
            f'insert into "LOG" ("TYPE_PK", "INFER_DATE", x, y, w, h, confidence, "QUESTION_PK", "PRED") \
                                values (\'{exam_info}\', \'{time}\', {x}, {y}, {w}, {h}, {confidence}, {pred}, {question_num});'
        )
        db.commit()
