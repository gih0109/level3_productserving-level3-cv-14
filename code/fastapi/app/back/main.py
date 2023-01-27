from pathlib import Path
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.param_functions import Depends
from pydantic import Json
from pdf2image import convert_from_bytes
from pycocotools.coco import COCO
from mmdet.apis import init_detector
import numpy as np
import sys
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

sys.path.append("/opt/ml/input/code/fastapi/app/back")
from inference import *
from utils import *


# settings
answer_dir = "/opt/ml/input/code/fastapi/app/answer"
model_config = "/opt/ml/input/data/models/19/config.py"
model_weight = "/opt/ml/input/data/models/19/model.pth"
coco = COCO("/opt/ml/input/data/annotations/train_v1-3.json")


app = FastAPI()

# 모델을 load 하는 부분입니다.
detector = init_detector(model_config, model_weight, device="cuda:0")

# 요청하는 시험에 대한 정답을 가져오는 부분입니다.
# TODO: 지훈님께서 만들어준 DB와 연결이 필요합니다.
Base = declarative_base()


class Answer(Base):
    __tablename__ = "answer"
    id = Column(Integer, primary_key=True, index=True)
    answer = Column(String, index=True)
    group = Column(String)


engine = create_engine("sqlite:///./answer.db")
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@app.post("/uploadfiles_name/{exam_info}")
def upload_csv_name(exam_info: str, csv_file: UploadFile = File(...)):
    try:
        df = pd.read_csv(csv_file.file)
        question = list(df["question"])
        answer = list(df["answer"])
        answer = list(map(str, answer))
        db = SessionLocal()
        for a in answer:
            db.add(Answer(answer=a, group=exam_info))
        db.commit()
        db.close()
        return {"question": question, "answers": answer}
    except:
        raise HTTPException(status_code=400, detail="Invalid CSV file")


@app.get("/answers/{exam_info}")
def get_answers(exam_info: str):
    db = SessionLocal()
    if db.query(Answer).filter(Answer.group == exam_info).count() > 0:
        return {"answers": None}
    else:
        return {"answers": "No data"}


def get_answers_from_db(exam_info):
    db = SessionLocal()
    answers = db.query(Answer).filter(Answer.group == exam_info).all()
    answer = {(i + 1): int(a.answer) for i, a in enumerate(answers)}
    db.close()
    return answer


# 이미지를 불러와서 모델 예측을 수행하는 부분입니다.
@app.post("/predict/{exam_info}")
def predict(exam_info: str, file: UploadFile = File(...)):
    answer = get_answers_from_db(exam_info)
    images = convert_from_bytes(file.file._file.read())
    images_np = [np.array(image) for image in images]
    inference = Inference(
        images=images_np,
        exam_info=exam_info,
        coco=coco,
        detector=detector,
    )
    result = inference.make_user_solution(True, True)
    _score = score(result, answer)

    scoring_img = inference.save_score_img(_score)
    imgByteArr = io.BytesIO()
    scoring_img[0].save(
        imgByteArr, save_all=True, append_images=scoring_img[1:], format="PDF"
    )
    return Response(imgByteArr.getvalue(), media_type="application/pdf")
