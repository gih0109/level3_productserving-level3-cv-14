from pathlib import Path  # 파일 경로 등을 관리하기 위해 사용합니다.
from pdf2image import convert_from_path  # 저장한 pdf 이미지를 불러오기 위해 사용합니다.
import requests  # backend와의 연결을 위해 필요합니다.
from PIL import Image  # jpg 이미지로 저장, 문제에 O, X등 패치를 붙이기 위해 필요합니다.
import albumentations as A  # image reshape을 위해 사용합니다.
import numpy as np  # image를 다룰때, PIL이미지로 변환할때 등 사용합니다.
from datetime import datetime

import streamlit as st  # frontend로 사용합니다.
from inference import *  # 문제와 정답(모델의 예측)을 매칭할때 사용합니다. [건혁님]
from fronted_function import *
from pycocotools.coco import COCO  # 바로위 모듈을 사용하기 위해 필요합니다.

st.set_page_config(layout="wide")


def main():
    st.title("몇점일까?")
    st.subheader("평가원 객관식 문제 자동채점 프로그램")

    # 사전에 학습된 모델 불러오기
    inference_model = make_inference_model(
        model_type="mmdetection",
        model_info=[
            "/opt/ml/input/code/work_dirs/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco.py",
            "/opt/ml/input/code/work_dirs/faster_rcnn_r50_fpn_fp16_1x_coco/best_bbox_mAP_epoch_90.pth",
        ],
        coco_json_path="/opt/ml/input/data/annotations/train_v1-3.json",
        img_folder_path="/opt/ml/input/code/fastapi/app/tmp",
    )

    # 시험 정보 및 서험지 받기
    year_choice, test_choice, type_choice = init_value()
    exam_choice = year_choice + "_" + test_choice + "_" + type_choice  # ex: 2021_f_a
    exam_info = exam_choice + "_0"  # 건혁님 모듈에 넣기 위한 input, TODO: 불필요하다면 "_0"은 제거
    read_info = requests.get(f"http://127.0.0.1:8001/{exam_choice}")
    answer = read_info.json()["answer"]
    uploaded_file = st.file_uploader("손으로 풀이된 시험지의 pdf파일을 업로드하세요.", type=["pdf"])

    if uploaded_file:
        imgs_path = Pdf2Jpg(uploaded_file).get_imgs_path()

        # model inference
        start_time = datetime.now()
        inference_state = st.text("Inference Answer Model...")
        inference_model.imgs_path = imgs_path
        inference_model.update_exam_info(exam_info)
        user_solution = inference_model.make_user_solution()
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        inference_state.text(f"Inference Answer Model...done! elapsed : {elapsed_time}")

        # 채점하는 모듈입니다. TODO는 위에 적어뒀습니다.
        start_time = datetime.now()
        scoring_state = st.text("Scoring...")
        scoring_result = score(user_solution, answer)
        st.write(scoring_result)
        show_score_img(scoring_result, inference_model, imgs_path)
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        scoring_state.text(f"Saving pdf...done! elapsed : {elapsed_time}")


main()


# TODO: 선택과목 대응 필요
# 2022년 부터는 확률과 통계, 미적분, 기하 등 과목을 선택할 수 있습니다.
# 이들 과목을 선택함에 따라 문제, 정답이 달라집니다.
# 이를 대응하기 위한 코드가 필요합니다. (아래는 이전에 작성된 예시 코드입니다.)
# if paper_choice == paper[3]:
#     st.write("2022-06월 모의고사")
#     type = ["확률과 통계", "미적분", "기하"]
#     type_choice = st.selectbox("유형을 선택해 주세요", type)
#     if type_choice == "확률과 통계":
#         read_info = requests.get("http://127.0.0.1:8001/2022-06/probability")
#         answer = read_info.json()["answer"]
#     elif type_choice == "미적분":
#         read_info = requests.get("http://127.0.0.1:8001/2022-06/calculus")
#         answer = read_info.json()["answer"]
#     elif type_choice == "기하":
#         read_info = requests.get("http://127.0.0.1:8001/2022-06/geometry")
#         answer = read_info.json()["answer"]
#     st.write("2022-06")
