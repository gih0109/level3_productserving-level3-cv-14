import streamlit as st
import requests
import sys
from stqdm import stqdm
import io
from streamlit_image_comparison import image_comparison
from argparse import ArgumentParser

sys.path.append("/opt/ml/input/code/fastapi/app/front")
from utils import *

st.set_page_config(
    page_title="몇 점 일 까 ?", layout="wide", initial_sidebar_state="expanded"
)

# css 설정
st.markdown(
    """<style>
.title{
  text-align: center;
  font-size: 75px;
  color: #343138;
  font-weight: bold;
  text-shadow: 4px 1px 1px gray;
}
.sub_title{
    text-align: center;
    font-size: 20px;
    color: #028DF7;
    margin-bottom: 50px;
    font-weight: bold;
}
.use{
    text-align:center;
    font-size: 50px;
    margin-top : 30px;
    font-weight: bold;
}
.preview{
    text-align:center;
    font-size: 50px;
    margin-top : 30px;
    margin-bottom : 50px;
    font-weight : bold;
}
.use1{
    font-size : 20px;
}
.use2{
    text-align:center;
    font-size: 40px;
    margin-top : 50px;
    font-weight: bold;
    margin-bottom: 30px:
}
.box{
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    box-shadow: 2px 2px 2px gray;
    cursor: pointer;
    text-align: center;
    margin-bottom: 20px;
    margin-top : 50px;
}
.main_use{
    text-align:center;
    font-size: 25px;
    margin-bottom : 20px;
    font-weight: bold;
    margin-bottom:30px;
    margin-top: 30px;
}
.explain{
    font-size: 20px;
    margin-bottom : 20px;
    margin-top : 20px;
}
.side_title{
    font-size : 40px;
    color:#636363;
    text-shadow: 1.5px 1.5px black;
    padding: 10px;
}
</style>""",
    unsafe_allow_html=True,
)
st.sidebar.markdown('<p class="side_title">AI 채점 서비스</p>', unsafe_allow_html=True)
st.sidebar.caption("보다 간편하게 시험지를 채점하세요")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
categories = ["Home", "Guideline", "채점하기"]
select = st.sidebar.selectbox("select a category", categories)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)


def home():
    st.markdown('<p class="title">몇 점 일 까 ?💯</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub_title">AI 채점 선생님이 당신을 대신해 채점해 드립니다</p>',
        unsafe_allow_html=True,
    )
    # st.markdown('<p class="preview">P r e v i e w</p>', unsafe_allow_html=True)
    st.markdown("<p> </p>", unsafe_allow_html=True)
    empty1, con, empty2 = st.columns([0.3, 0.9, 0.3])
    with empty1:
        st.empty()
    with con:
        image_comparison(
            img1="/opt/ml/input/code/fastapi/app/front/explain_img/solve.jpg",
            img2="/opt/ml/input/code/fastapi/app/front/explain_img/check.jpg",
            label1="제출된 시험지",
            label2="AI 채점 결과",
        )
    with empty2:
        st.empty()


def introduce():
    # st.markdown('<p class="title">몇 점 일 까 ?💯</p>', unsafe_allow_html=True)
    # st.markdown(
    #     '<p class="sub_title">AI 채점 선생님이 당신을 대신해 채점해 드립니다</p>',
    #     unsafe_allow_html=True,
    # )
    # _, con1, con2, _ = st.columns([0.1, 0.2, 0.1, 0.1])
    # _, con1, con2, _ = st.columns([0.15, 0.55, 0.1, 0.1])
    # Guideline = con1.button("**Guideline**")
    # warning = con2.button("**주의사항**")
    # st.markdown("<hr>", unsafe_allow_html=True)
    # if Guideline:
    st.markdown('<p class="use">G u i d e l i n e</p>', unsafe_allow_html=True)
    user, res = st.columns([0.5, 0.5])
    with user:
        st.markdown(
            '<div class = "box"><h5 class="use1">How to Use</h5></div>',
            unsafe_allow_html=True,
        )
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/use_intro.jpg",
            width=600,
            caption="다음 화면에 시험지를 올려주세요",
        )
        st.markdown("<p class='explain'>.</p>", unsafe_allow_html=True)
        st.markdown("<p class='explain'>.</p>", unsafe_allow_html=True)
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/res.jpg",
            width=600,
            caption="다운로드 버튼을 클릭",
        )
    with res:
        st.markdown(
            '<div class = "box"><h5 class="use1">Grading Result</h5></div>',
            unsafe_allow_html=True,
        )
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/scoring.jpg",
            width=580,
            caption="채점된 결과",
        )
        # st.markdown("<hr>", unsafe_allow_html=True)
        # st.markdown('<p class="preview">P r e v i e w</p>', unsafe_allow_html=True)
        # st.markdown("<p> </p>", unsafe_allow_html=True)
        # empty1, con, empty2 = st.columns([0.3, 0.8, 0.3])
        # with empty1:
        #     st.empty()
        # with con:
        #     image_comparison(
        #         img1="/opt/ml/input/code/fastapi/app/front/explain_img/solve.jpg",
        #         img2="/opt/ml/input/code/fastapi/app/front/explain_img/check.jpg",
        #         label1="제출된 시험지",
        #         label2="Ai채점 결과",
        #     )
        # with empty2:
        #     st.empty()
    # if warning:
    st.markdown('<p class="use2">객 관 식</p>', unsafe_allow_html=True)
    war_cor_1, war_incor_1 = st.columns([0.5, 0.5])
    with war_cor_1:
        st.markdown(
            '<div class = "box"><h5 class="use2_1">올바른 방법</h5></div>',
            unsafe_allow_html=True,
        )
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/y.jpg",
            width=620,
            caption='정답 선지에만 "V" 표시를 해주세요',
        )
    with war_incor_1:
        st.markdown(
            '<div class = "box"><h5 class="use1">틀린 방법</h5></div>',
            unsafe_allow_html=True,
        )
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/n1.jpg",
            width=620,
            caption='정답을 "O"로 표시하지 말아주세요',
        )
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/n2.jpg",
            width=620,
            caption='틀린 선지에 "X" 또는 "/" 표시를 하지 말아주세요',
        )
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/n3.jpg",
            width=620,
            caption="정답을 선지 번호로 작성하지 말아주세요",
        )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="use2">주 관 식</p>', unsafe_allow_html=True)
    war_cor_2, war_incor_2 = st.columns([0.5, 0.5])
    with war_cor_2:
        st.markdown(
            '<div class = "box"><h5 class="use2_1">올바른 방법</h5></div>',
            unsafe_allow_html=True,
        )
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/black_square.jpg",
            width=580,
            caption="정답에 네모박스를 그려주세요",
        )
    with war_incor_2:
        st.markdown(
            '<div class = "box"><h5 class="use1">틀린 방법</h5></div>',
            unsafe_allow_html=True,
        )
        st.image(
            "/opt/ml/input/code/fastapi/app/front/explain_img/black.jpg",
            width=580,
            caption="정답만 있는 경우를 주의해 주세요",
        )


def main():
    st.markdown('<p class="title">몇 점 일 까 ?💯</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub_title">AI 채점 선생님이 당신을 대신해 채점해 드립니다</p>', unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    # st.markdown('<p class="main_use">평가원 객관식 문제 자동채점 프로그램</p>', unsafe_allow_html=True)
    # st.title("몇점일까?")
    st.subheader("평가원 문제 자동채점 프로그램")
    # 스트림릿의 선택 창으로 채점할 문제의 종류를 선택하고, 정답지를 불러오는 부분입니다.
    year_choice, test_choice, type_choice = init_value()
    exam_info = year_choice + "_" + test_choice + "_" + type_choice  # ex: 2021_f_a
    a = [1, 2]
    # response = requests.get(f"http://{backend_server}/answers/{exam_info}")
    # rs = response.json()["answers"]
    # if rs == "No data":
    #     file = st.file_uploader("정답 데이터가 없습니다, 답안을 등록해주세요", type=["csv"])
    #     if file:
    #         csv_file = file.read()
    #         response = requests.post(
    #             f"http://{backend_server}/uploadfiles_name/{exam_info}",
    #             files={"csv_file": csv_file},
    #         )
    #         st.write("등록이 완료되었습니다.")
    uploaded_file = st.file_uploader("손으로 풀이된 시험지의 pdf파일을 업로드하세요.", type=["pdf"])
    if uploaded_file:
        # 업로드한 파일을 backend server에 보내서 모델 예측을 받는 부분입니다.
        # length = 1  # TODO: uploaded_file의 길이로 수정합니다.
        files = {"file": uploaded_file.getvalue()}
        # progress = stqdm(total=length)
        user_solution = requests.post(
            f"http://{backend_server}/predict/{exam_info}", files=files
        )
        # progress.update(1)
        st.download_button(
            "Download Scored Image",
            data=io.BytesIO(user_solution.content).read(),
            file_name="scoring.pdf",
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    if select == "Guideline":
        introduce()
    elif select == "채점하기":
        parser = ArgumentParser()
        parser.add_argument("--BackendServer", type=str, default="34.64.169.3:30002")
        args = parser.parse_args()
        main(args)
    elif select == "Home":
        home()
