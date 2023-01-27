import streamlit as st
import PIL.Image as Image


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


def show_score_img(scoring_result, inference_model, imgs_path):
    # 채점된 이미지를 만들기 위해 o, x 이미지를 불러오는 부분입니다.
    # TODO: 위의 input이미지의 resize 부분과 함께 고려해야 할 사항입니다.
    o_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/correct.png")
    x_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/wrong.png")
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
                        int(bbox[1] - o_height / 2) + 10,
                    ),
                    o_image,
                )
            elif scoring_result[question] == "X":
                background.paste(
                    x_image,
                    (
                        int(bbox[0] - x_width / 2),
                        int(bbox[1] - x_height / 2) + 10,
                    ),
                    x_image,
                )
        st.image(np.array(background))
