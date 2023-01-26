from pathlib import Path
import pandas as pd
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.param_functions import Depends
from pydantic import Json
from pdf2image import convert_from_bytes
from pycocotools.coco import COCO
from mmdet.apis import init_detector
import numpy as np
import sys

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
@app.get("/answer/{exam_id}", description="해당 시험에 대한 정답을 가져옵니다")
async def get_answers(exam_id: str) -> dict:
    answer_path = Path(answer_dir, exam_id + ".csv")
    answer = pd.read_csv(answer_path)
    answer.index += 1  # 1번부터 시작하도록
    parsed = json.loads(answer.to_json(orient="columns"))
    return parsed


# 이미지를 불러와서 모델 예측을 수행하는 부분입니다.
@app.post("/predict/{exam_info}")
def predict(exam_info: str, file: UploadFile = File(...)):
    images = convert_from_bytes(file.file._file.read())
    images_np = [np.array(image) for image in images]
    inference = Inference(
        images=images_np,
        exam_info="2021_f_a",
        coco=coco,
        detector=detector,
    )
    result = inference.make_user_solution()
    return result


# 모델 예측과 정답을 받아서 채점을 하는 부분입니다.
@app.post("/score")
async def get_score(user_solution: Json = Form(...), answer: Json = Form(...)):
    return score(user_solution, answer)
