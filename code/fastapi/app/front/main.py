import streamlit as st
import json
import requests
import sys

sys.path.append("/opt/ml/input/code/fastapi/app/front")
from utils import *

backend_server = "49.50.174.162:30002"

st.set_page_config(layout="wide")


def main():
    st.title("몇점일까?")
    st.subheader("평가원 객관식 문제 자동채점 프로그램")

    # 스트림릿의 선택 창으로 채점할 문제의 종류를 선택하고, 정답지를 불러오는 부분입니다.
    year_choice, test_choice, type_choice = init_value()
    exam_info = year_choice + "_" + test_choice + "_" + type_choice  # ex: 2021_f_a
    answer_json = requests.get(
        f"http://{backend_server}/answer/{exam_info}"
    ).json()  # columns ("문항번호", "정답", "배점")

    uploaded_file = st.file_uploader("손으로 풀이된 시험지의 pdf파일을 업로드하세요.", type=["pdf"])
    if uploaded_file:
        # 업로드한 파일을 backend server에 보내서 모델 예측을 받는 부분입니다.
        files = {"file": uploaded_file.getvalue()}
        user_solution = requests.post(
            f"http://{backend_server}/predict/{exam_info}", files=files
        ).json()

        # 모델 예측과 정답을 보내서 채점을 하는 부분입니다.
        scoring_data = {
            "user_solution": json.dumps(user_solution),
            "answer": json.dumps(answer_json["정답"]),
        }
        scoring_result = requests.post(
            f"http://{backend_server}/score", data=scoring_data
        )
        st.write(scoring_result.json())

        # TODO: 채점 결과를 이미지로 반환합니다.


if __name__ == "__main__":
    main()
