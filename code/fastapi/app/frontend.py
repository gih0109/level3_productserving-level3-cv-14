from pathlib import Path  # 파일 경로 등을 관리하기 위해 사용합니다.
from pdf2image import convert_from_path  # 저장한 pdf 이미지를 불러오기 위해 사용합니다.
import requests  # backend와의 연결을 위해 필요합니다.
from PIL import Image  # jpg 이미지로 저장, 문제에 O, X등 패치를 붙이기 위해 필요합니다.
import albumentations as A  # image reshape을 위해 사용합니다.
import numpy as np  # image를 다룰때, PIL이미지로 변환할때 등 사용합니다.

import streamlit as st  # frontend로 사용합니다.
from inference_test import make_qadict  # 문제와 정답(모델의 예측)을 매칭할때 사용합니다. [건혁님]

st.set_page_config(layout="wide")

# 채점하는 함수
# TODO: type 통일이 필요합니다.
# answer의 경우, key, value가 모두 str 타입인데, user_solution은 int타입이라 불필요한 변환과정이 들어갑니다.
# user_solution dictionary의 key, value도 모두 str로 통일해서 불필요한 타입 변환을 줄이면 좋을 것 같습니다.
def score(user_solution: dict, answer: dict) -> dict:
    user_solution = {f"{k}": f"{v}" for k, v in user_solution.items()}
    result = {}
    # TODO: 경우에 따라 유연하게 객관식 문제 모두를 대처할 수 있도록 수정이 필요합니다.
    # user_solution dictionary가 객관식 문제만 포함하고, answer는 1~30번까지 모두 존재해서 indexing error를 방지하고자,
    # 1부터 21번까지만 수행하게끔 하드코딩 되어 있습니다.
    for question in map(str, range(1, 22)):
        if user_solution[question] == answer[question]:
            result[question] = "O"
        else:
            result[question] = "X"
    return result


def main():
    st.title("몇점일까?")
    st.subheader("평가원 객관식 문제 자동채점 프로그램")

    # 스트림릿의 선택 창으로 채점할 문제의 종류를 선택하는 부분입니다.
    year = [str(y) for y in range(2013, 2024)]  # 2013학년도 ~ 2023학년도
    year_choice = st.selectbox("채점을 원하시는 시험의 연도를 선택해 주세요", year)

    test = ["6월", "9월", "수능"]  # 6월, 9월 모의고사, 수능입니다.
    test_map = {"6월": "6", "9월": "9", "수능": "f"}  # 데이터에 사용된 문자(6, 9, f)로 변환하기 위한 맵입니다.
    test_choice = st.selectbox("채점을 원하시는 시험을 선택해 주세요", test)

    type_ = (
        ["구분없음"] if int(year_choice) >= 2022 else ["가(A)형", "나(B)형"]
    )  # 가(A)형, 나(B)형, 2022학년도부터 구분 없음(n)
    type_map = {"구분없음": "n", "가(A)형": "a", "나(B)형": "b"}
    type_choice = st.selectbox("채점을 원하시는 시험의 종류를 선택해 주세요", type_)

    # TODO: 홀수형, 짝수형 구분도 필요합니다.
    # 현재 수집된 데이터는 홀수형만 있는 것 같습니다. 정답이 상이할 것으로 생각됩니다.

    exam_choice = (
        year_choice + "_" + test_map[test_choice] + "_" + type_map[type_choice]
    )  # ex: 2021_f_a
    exam_info = exam_choice + "_0"  # 건혁님 모듈에 넣기 위한 input, TODO: 불필요하다면 "_0"은 제거

    read_info = requests.get(f"http://127.0.0.1:8001/{exam_choice}")
    answer = read_info.json()["answer"]

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

    uploaded_file = st.file_uploader("손으로 풀이된 시험지의 pdf파일을 업로드하세요.", type=["pdf"])

    if uploaded_file:
        # pdf 이미지를 tmp 폴더에 저장하는 코드입니다.
        # TODO: 현재는 aistages 서버의 고정된 공간에 저장하도록 되어 있지만, 향후 DB 연결 등으로 확장 할 수 있습니다.
        # 이경우 user_id등을 부여해서 여러명의 사용자가 동시 접속한 경우에도 대응할 수 있도록 할 수 있습니다.
        # 다만 그러면 로그인 기능 등을 추가로 구현해야 합니다.
        save_path = Path("/opt/ml/input/code/fastapi/app/tmp", uploaded_file.name)
        with open(save_path, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        # tmp 폴더에 저장한 pdf 이미지 불러오고 resize 합니다.
        # TODO: resize 없이 원본이미지를 그대로 활용할 수도 있습니다. (annotation 정보를 역으로 이미지 크기에 맞춤)
        # 현재 annotation한 question box의 좌표정보를 이용하기 위해서는 우리가 hasty에 올렸던 이미지 크기(3309, 2339)와 동일해야 합니다.
        # 불필요한 resize를 줄인다면 더욱 시간을 절약할 수 있습니다.
        images = convert_from_path(save_path)
        for idx, image in enumerate(images):
            images[idx] = A.resize(np.array(image), 3309, 2339)

        # TODO: 아래 코드는 건혁님 모듈과 연결하기 위한 불필요한 작업입니다. 제거하고 바로 연결할 수 있도록 수정이 필요합니다.
        # 건혁님 모듈에 연결하기 위해서 argument 생성 (imgs_path, jpg 이미지로 나눠서 다시 tmp에 저장)
        imgs_path = []
        for idx, image in enumerate(images):
            pil_image = Image.fromarray(image)
            save_path = "/opt/ml/input/code/fastapi/app/tmp/" + str(idx + 1) + ".jpg"
            pil_image.save(save_path, "JPEG")
            save_path = str(idx + 1) + ".jpg"
            imgs_path.append(save_path)

        # 지정된 위치의 모델을 불러오고, 사용자 답안에 대한 예측을 수행한 뒤 문제박스와 답안박스에 대한 mIOU기반으로 문제와 답안을 매칭합니다.
        # return 형태 : {q : a}
        # TODO: 가장 핵심적인 부분으로 모델 예측 성능도 향상시켜야 하며, 문제와 답안을 매칭하는 알고리즘도 정교하게 발전시켜야 합니다.
        # 현재는 건혁님이 코드를 작성해 주셨습니다.
        # GCP 배포를 고려한다면 추후에 mmdeploy의 onnx 모델(cpu)을 사용하거나,
        # 다른 서버에서 gpu를 사용한 예측만 수행해서 결과를 받아오는 방식으로 수정할 수 있습니다.
        user_solution = make_qadict(
            model_type="mmdetection",
            model_info=[
                "/opt/ml/input/code/fastapi/app/model/cascade_rcnn_pafpn_convnext-xl_last.py",
                "/opt/ml/input/code/fastapi/app/model/best_bbox_mAP_epoch_15.pth",
            ],
            coco_json_path="/opt/ml/input/data/annotations/train_v1-3.json",
            img_folder_path="/opt/ml/input/code/fastapi/app/tmp",
            imgs_path=imgs_path,
            exam_info=exam_info,
        )

        # 채점하는 모듈입니다. TODO는 위에 적어뒀습니다.
        scoring_result = score(user_solution=user_solution, answer=answer)

        st.write(scoring_result)


main()
