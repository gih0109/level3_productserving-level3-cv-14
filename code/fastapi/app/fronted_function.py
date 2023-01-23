from pathlib import Path  # 파일 경로 등을 관리하기 위해 사용합니다.
from pdf2image import convert_from_path  # 저장한 pdf 이미지를 불러오기 위해 사용합니다.
from datetime import datetime
from PIL import Image  # jpg 이미지로 저장, 문제에 O, X등 패치를 붙이기 위해 필요합니다.
import streamlit as st  # frontend로 사용합니다.
import numpy as np  # image를 다룰때, PIL이미지로 변환할때 등 사용합니다.
import albumentations as A  # image reshape을 위해 사용합니다.


class Pdf2Jpg:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def get_imgs_path(self):
        # pdf 이미지를 tmp 폴더에 저장하는 코드입니다.
        # TODO: 현재는 aistages 서버의 고정된 공간에 저장하도록 되어 있지만, 향후 DB 연결 등으로 확장 할 수 있습니다.
        # 이경우 user_id등을 부여해서 여러명의 사용자가 동시 접속한 경우에도 대응할 수 있도록 할 수 있습니다.
        # 다만 그러면 로그인 기능 등을 추가로 구현해야 합니다.
        start_time = datetime.now()
        file_write_state = st.text("Writing pdf...")
        save_path = Path("/opt/ml/input/code/fastapi/app/tmp", self.uploaded_file.name)
        with open(save_path, mode="wb") as w:
            w.write(self.uploaded_file.getvalue())
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        file_write_state.text(f"Writing pdf...done! elapsed : {elapsed_time}")

        # tmp 폴더에 저장한 pdf 이미지 불러오고 resize 합니다.
        # TODO: resize 없이 원본이미지를 그대로 활용할 수도 있습니다. (annotation 정보를 역으로 이미지 크기에 맞춤)
        # 현재 annotation한 question box의 좌표정보를 이용하기 위해서는 우리가 hasty에 올렸던 이미지 크기(3309, 2339)와 동일해야 합니다.
        # 불필요한 resize를 줄인다면 더욱 시간을 절약할 수 있습니다.
        start_time = datetime.now()
        file_write_state = st.text("Reshape imgs...")
        images = convert_from_path(save_path)
        for idx, image in enumerate(images):
            images[idx] = np.array(image)
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        file_write_state.text(f"Reshaping imgs...done! elapsed : {elapsed_time}")

        # TODO: 아래 코드는 건혁님 모듈과 연결하기 위한 불필요한 작업입니다. 제거하고 바로 연결할 수 있도록 수정이 필요합니다.
        # 건혁님 모듈에 연결하기 위해서 argument 생성 (imgs_path, jpg 이미지로 나눠서 다시 tmp에 저장)
        imgs_path = []
        for idx, image in enumerate(images):
            pil_image = Image.fromarray(image)
            save_path = "/opt/ml/input/code/fastapi/app/tmp/" + str(idx + 1) + ".jpg"
            pil_image.save(save_path, "JPEG")
            save_path = str(idx + 1) + ".jpg"
            imgs_path.append(save_path)

        return imgs_path


def show_score_img(scoring_result, inference_model, imgs_path):
    # 채점된 이미지를 만들기 위해 o, x 이미지를 불러오는 부분입니다.
    # TODO: 위의 input이미지의 resize 부분과 함께 고려해야 할 사항입니다.
    o_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_img/correct.png")
    x_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_img/wrong.png")
    o_width, o_height = o_image.size
    x_width, x_height = x_image.size

    exam_info = inference_model.exam_info

    # TODO: 현재 paste 좌표가 좌측 하단으로 잡혀있음 (좌측 상단으로 바꿔야함. annotation 정보 확인 필요)
    for img in imgs_path:  # fix
        background = Image.open(f"/opt/ml/input/code/fastapi/app/tmp/{img}").convert(
            "RGBA"
        )  # 배경 이미지 생성
        question_ann = inference_model.load_anns_q(
            exam_info, img, inference_model.coco
        )  # 이미지에 대한 question annotation 정보 획득
        for cat_id, bbox in question_ann.items():
            question = str(cat_id - 6)  # 문제 번호: 1 ~ 30
            if scoring_result[question] == "O":
                background.paste(
                    o_image,
                    (
                        int(bbox[0] - o_width / 2),
                        int(bbox[1] - o_height / 2),
                    ),
                    o_image,
                )
            else:
                background.paste(
                    x_image,
                    (
                        int(bbox[0] - x_width / 2),
                        int(bbox[1] - x_height / 2),
                    ),
                    x_image,
                )
        st.image(np.array(background))


def init_value():
    # 스트림릿의 선택 창으로 채점할 문제의 종류를 선택하는 부분입니다.
    year = [str(y) for y in range(2013, 2024)]  # 2013학년도 ~ 2023학년도
    default_ix = year.index("2021")
    year_choice = st.selectbox("채점을 원하시는 시험의 연도를 선택해 주세요", year, index=default_ix)

    test = ["6월", "9월", "수능"]  # 6월, 9월 모의고사, 수능입니다.
    test_map = {"6월": "6", "9월": "9", "수능": "f"}  # 데이터에 사용된 문자(6, 9, f)로 변환하기 위한 맵입니다.
    test_choice = st.selectbox("채점을 원하시는 시험을 선택해 주세요", test, index=2)

    type_ = (
        ["구분없음"] if int(year_choice) >= 2022 else ["가(A)형", "나(B)형"]
    )  # 가(A)형, 나(B)형, 2022학년도부터 구분 없음(n)
    type_map = {"구분없음": "n", "가(A)형": "a", "나(B)형": "b"}
    type_choice = st.selectbox("채점을 원하시는 시험의 종류를 선택해 주세요", type_)

    return year_choice, test_map[test_choice], type_map[type_choice]


def score(user_solution=None, answer=None):
    """채점하는 함수
    answer의 경우, key, value가 모두 str 타입인데, user_solution은 int타입이라 불필요한 변환과정이 들어갑니다.
    user_solution dictionary의 key, value도 모두 str로 통일해서 불필요한 타입 변환을 줄이면 좋을 것 같습니다.

    Args:
        user_solution (dict): _description_. Defaults to None.
        answer (dict): _description_. Defaults to None.

    Returns:
        dict: key : 문제번호, value : O or X
    """
    user_solution = {f"{k}": f"{v}" for k, v in user_solution.items()}
    result = {}
    # TODO: 경우에 따라 유연하게 객관식 문제 모두를 대처할 수 있도록 수정이 필요합니다.
    # user_solution dictionary가 객관식 문제만 포함하고, answer는 1~30번까지 모두 존재해서 indexing error를 방지하고자,
    # 1부터 21번까지만 수행하게끔 하드코딩 되어 있습니다.
    for question in map(str, range(1, 22)):  # fix
        if user_solution[question] == answer[question]:
            result[question] = "O"
        else:
            result[question] = "X"
    return result
