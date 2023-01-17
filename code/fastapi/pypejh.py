import streamlit as st
import numpy as np
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from pathlib import Path
import mmcv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from pdf2image import convert_from_path
from datetime import datetime
import pandas as pd
import albumentations as A
import pickle

# 모델을 불러올때 사용하는 함수입니다.
@st.cache(allow_output_mutation=True)
def load_model(config, checkpint, device):
    return init_detector(config, checkpint, device=device)


def is_empty(answer_preds):
    num_element = sum([answer.size for answer in answer_preds])
    return True if num_element == 0 else False


def match_QandA(question_preds, answer_preds, len_images):
    question_answer_set = []
    for page in range(len_images):
        for question_number, question_box in enumerate(question_preds[page]):
            if question_box.size != 0:
                question_left_bottom = question_box[0][[0, 3]]  # 좌측 하단
                min_dist = 999  # initialize
                if is_empty(answer_preds[page]):  # 객관식 문제가 없는 경우 스킵
                    continue
                # 고정된 문제에 대해서 모든 user box와 거리 계산 후 가까운 box 매칭
                for user_answer, answer_box in enumerate(answer_preds[page]):
                    if answer_box.size != 0:
                        for answer in answer_box:  # 1번으로 체크한 경우가 여러개 일 수 있음.
                            user_answer_left_bottom = answer[[0, 3]]  # 좌측 하단
                            # 유클리드 거리 계산후 가까우면 업데이트
                            euclidian_dist = np.linalg.norm(
                                question_left_bottom - user_answer_left_bottom
                            )
                            if euclidian_dist < min_dist:
                                min_dist = euclidian_dist
                                matched = (
                                    page + 1,
                                    question_number + 1,
                                    user_answer + 1,
                                    question_box[0],
                                    answer,
                                )
                if (
                    question_answer_set
                    and question_answer_set[-1][0] == matched[0]
                    and all(question_answer_set[-1][4] == matched[4])
                ):  # 가장 마지막에 추가한 쌍과 페이지가 같고, answer box도 같다면 스킵
                    pass
                else:
                    question_answer_set.append(matched)
    return question_answer_set


def mark(question_answer_set, answer, images, o_image, x_image):
    o_width, o_height = o_image.size
    x_width, x_height = x_image.size

    score = 0
    page = -1
    background = None
    after_scoring = []
    for matched in question_answer_set:
        if matched[0] != page:  # 새로운 페이지로 넘어가면
            if background is not None:
                after_scoring.append(background.convert("RGB"))
            page = matched[0]  # 페이지 업데이트
            background = Image.fromarray(np.uint8(images[page - 1])).convert(
                "RGBA"
            )  # 배경 이미지 생성
        question_number = matched[1]
        user_answer = matched[2]
        if user_answer == answer["정답"][answer["문항번호"] == question_number].item():
            # 이미지에 채점
            background.paste(
                o_image,
                (
                    int(matched[3][0] - o_width / 2),
                    int(matched[3][1] - o_height / 2) + 10,
                ),
                o_image,
            )
            score += answer["배점"][answer["문항번호"] == question_number].item()
        else:
            background.paste(
                x_image,
                (
                    int(matched[3][0] - x_width / 2),
                    int(matched[3][1] - x_height / 2) + 10,
                ),
                x_image,
            )
    after_scoring.append(background.convert("RGB"))
    return after_scoring, score


def main(
    save_folder,  # 사용자가 업로드한 pdf 파일을 임시로 서버에 저장할 위치(str) 입니다.
    config_question,  # 문제가 몇번인지 찾아주는 모델의 config(.py) 입니다.
    checkpoint_question,  # 문제가 몇번인지 찾아주는 모델의 weight(.pth) 입니다.
    config_answer,  # 사용자가 선택한 답안이 몇번인지 찾아주는 모델의 config(.py) 입니다.
    checkpoint_answer,  # 사용자가 선택한 답안이 몇번인지 찾아주는 모델의 weight(.pth) 입니다.
    resize_shape,  # 이미지 reshape에 적용할 크기 입니다. (h, w)
    threshold=0.5,  # 모델의 예측을 걸러낼때 사용할 threshold 값입니다. (ex. confidence score가 0.5 이상인 예측만 사용)
):
    start_time = datetime.now()
    model_load_state = st.text("Loading model, page...")
    # TODO: question model loading 구현 필요 (모델이 준비되면 아래 주석 해제)
    # if "question_model" not in st.session_state:
    #     st.session_state.question_model = load_model(
    #         config_question, checkpoint_question, "cuda"
    #     )
    if "answer_model" not in st.session_state:
        st.session_state.answer_model = load_model(
            config_answer, checkpoint_answer, "cuda"
        )
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    model_load_state.text(f"Loading model...done! elapsed : {elapsed_time}")

    # Page Loading
    # TODO: 사용자가 연도, 수능/모의고사, 6/9월, 가/나형 선택하도록 layout 변경해야함
    start_time = datetime.now()
    page_load_state = st.text("Loading page...")
    st.title("Prototype")
    uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    page_load_state.text(f"Loading page...done! elapsed : {elapsed_time}")

    if uploaded_file is not None:
        # TODO: File을 서버에 Save 하지 않고, 바로 사용할 수 있도록 고치면 좋음
        # File Upload & Save
        start_time = datetime.now()
        file_write_state = st.text("Writing pdf...")
        save_path = Path(save_folder, "input", uploaded_file.name)
        with open(save_path, mode="wb") as w:
            w.write(uploaded_file.getvalue())
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        file_write_state.text(f"Writing pdf...done! elapsed : {elapsed_time}")

        # Read Pdf as image
        start_time = datetime.now()
        image_read_state = st.text("Loading pdf...")
        images = convert_from_path(save_path)
        for idx, image in enumerate(images):
            images[idx] = A.resize(np.array(image), resize_shape[0], resize_shape[1])
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        image_read_state.text(f"Loading pdf...done! elapsed : {elapsed_time}")

        # Read Answer sheet
        start_time = datetime.now()
        answer_read_state = st.text("Loading Answersheet...")
        # TODO: 사용자가 layout에서 연도, 수능/모의고사, 6/9월, 가/나형 선택하면, 이에 맞는 answer를 찾아서 불러올 수 있도록 수정해야함.
        answer_path = Path(
            save_folder,
            "answer",
            "2020학년도 대학수학능력시험 6월 모의평가(가형).csv",
        )
        answer = pd.read_csv(answer_path)
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        answer_read_state.text(f"Loading Answersheet...done! elapsed : {elapsed_time}")

        # Setting model predictions
        # TODO: class를 여기서 지정하지 않고, 모델 config에서 바로 찾아올 수 있도록 고쳐야함.
        question_classes = (
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
            "q13",
            "q14",
            "q15",
            "q16",
            "q17",
            "q18",
            "q19",
            "q20",
            "q21",
            "q22",
            "q23",
            "q24",
            "q25",
            "q26",
            "q27",
            "q28",
            "q29",
            "q30",
        )
        question_cat_idxs = range(0, len(question_classes))
        answer_classes = ("a1", "a2", "a3", "a4", "a5")
        answer_cat_idxs = range(0, len(answer_classes))

        # Model Inference (Question Number)
        # question_preds[page][category][pred]
        # question_preds[0]: 첫번째 페이지에 대한 예측 (len: 30)
        # question_preds[0][0]: 첫번째 페이지에서 첫번째 카테고리(q1)에 대한 예측 (ndarray: (n ,4)) n: 예측 갯수
        # question_preds[0][0][0] -> (xmin, ymin, xmax, ymax)
        start_time = datetime.now()
        image_read_state = st.text("Inference Question Model...")
        question_preds = []
        # for image in images:
        #     full_inference = inference_detector(st.session_state.answer_model, image)
        #     # confidence score가 thresold 이상만 filter
        #     question_preds.append(
        #         [
        #             full_inference[cat_idx][full_inference[cat_idx][:, 4] >= threshold][
        #                 :, :-1
        #             ]
        #             for cat_idx in question_cat_idxs
        #         ]
        #     )
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        image_read_state.text(
            f"Inference Question Model...done! elapsed : {elapsed_time}"
        )

        # Model Inference (User Answer)
        # TODO: "a0" catetory를 포함해서 class가 6개가 되도록 변경해야함. (annotation, model, classes)
        # answer_preds[page][category][pred]
        # answer_preds[0]: 첫번째 페이지에 대한 예측 (len: 6)
        # answer_preds[0][0]: 첫번째 페이지에서 첫번째 카테고리(a1)에 대한 예측 (ndarray: (n ,4)) n: 예측 갯수
        # answer_preds[0][0][0] -> (xmin, ymin, xmax, ymax)
        start_time = datetime.now()
        image_read_state = st.text("Inference Answer Model...")
        answer_preds = []
        for image in images:
            full_inference = inference_detector(st.session_state.answer_model, image)
            # confidence score가 thresold 이상만 filter
            answer_preds.append(
                [
                    full_inference[cat_idx][full_inference[cat_idx][:, 4] >= threshold][
                        :, :-1
                    ]
                    for cat_idx in answer_cat_idxs
                ]
            )
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        image_read_state.text(
            f"Inference Answer Model...done! elapsed : {elapsed_time}"
        )

        # Model output 후처리 (완벽한 답안이 나오도록)
        # TODO: 후처리 모듈 추가, 우선은 사람이 만든 답안으로 대체
        with open("/opt/ml/ASS/data/temp/answer_preds.pickle", "rb") as f:
            answer_preds = pickle.load(f)
        with open("/opt/ml/ASS/data/temp/question_preds.pickle", "rb") as f:
            question_preds = pickle.load(f)

        # Matching Question and Answer
        start_time = datetime.now()
        matching_state = st.text("Matching Question and Answer...")
        question_answer_set = match_QandA(question_preds, answer_preds, len(images))
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        matching_state.text(
            f"Matching Question and Answer...done! elapsed : {elapsed_time}"
        )

        # Scoring
        # o, x 이미지 파일 불러오기
        o_image = Image.open("/opt/ml/ASS/data/correct.png")
        o_image = o_image.resize((150, 150))
        x_image = Image.open("/opt/ml/ASS/data/wrong.png")
        x_image = x_image.resize((100, 100))

        start_time = datetime.now()
        matching_state = st.text("Scoring...")
        after_scoring, score = mark(
            question_answer_set, answer, images, o_image, x_image
        )
        # 채점하지 못한 페이지가 생기면 뒤에 추가
        if len(after_scoring) != len(images):
            diff = len(images) - len(after_scoring)
            after_scoring.extend(
                [
                    Image.fromarray(np.uint8(images[len(after_scoring) + i])).convert(
                        "RGB"
                    )
                    for i in range(diff)
                ]
            )
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        matching_state.text(f"Scoring...done! elapsed : {elapsed_time}")

        # Convert to pdf
        start_time = datetime.now()
        saving_pdf_state = st.text("Saving pdf...")
        after_scoring[0].save(
            "/opt/ml/ASS/data/temp/scored.pdf",
            save_all=True,
            append_images=after_scoring[1:],
        )
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        saving_pdf_state.text(f"Saving pdf...done! elapsed : {elapsed_time}")

        # Enable Saving
        with open("/opt/ml/ASS/data/temp/scored.pdf", "rb") as file:
            btn = st.download_button(
                label="click me to download pdf",
                data=file,
                file_name="scored.pdf",
                mime="application/octet-stream",
            )


if __name__ == "__main__":
    config_question = "/opt/ml/ASS/data/model_config.py"
    checkpoint_question = "/opt/ml/ASS/data/model_weight.pth"
    config_answer = "/opt/ml/ASS/data/answer_model_config.py"
    checkpoint_answer = "/opt/ml/ASS/data/answer_model_weight.pth"
    save_folder = "/opt/ml/ASS/data/"
    main(
        save_folder,  # 사용자가 업로드한 pdf 파일을 임시로 서버에 저장할 위치(str) 입니다.
        config_question,  # 문제가 몇번인지 찾아주는 모델의 config(.py) 입니다.
        checkpoint_question,  # 문제가 몇번인지 찾아주는 모델의 weight(.pth) 입니다.
        config_answer,  # 사용자가 선택한 답안이 몇번인지 찾아주는 모델의 config(.py) 입니다.
        checkpoint_answer,  # 사용자가 선택한 답안이 몇번인지 찾아주는 모델의 weight(.pth) 입니다.
        resize_shape=(
            1024,
            724,
        ),  # 이미지 reshape에 적용할 크기 입니다. (h, w)
        threshold=0.5,  # 모델의 예측을 걸러낼때 사용할 threshold 값입니다. (ex. confidence score가 0.5 이상인 예측만 사용)
    )
