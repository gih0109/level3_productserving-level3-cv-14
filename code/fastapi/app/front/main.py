import streamlit as st
import requests
import sys
from stqdm import stqdm
import io
from streamlit_image_comparison import image_comparison

sys.path.append("/opt/ml/input/code/fastapi/app/front")
from utils import *

backend_server = "0.0.0.0:30002"

st.set_page_config(layout="wide")


categories = ['about us','Guideline', '체점하기']
select = st.sidebar.selectbox("select a category", categories)


def explain(): # 사용 설명서 

    st.title("꼭 읽어주세요")
    st.markdown("<h3>시험 문제 풀 때 주의 사항</h3>", unsafe_allow_html=True)
    mult_correct_button = st.button("객관식")
    correct_button = st.button("주관식")
    st.write(" ")
    st.write(" ")
    if mult_correct_button: 

        row1, row2 = st.columns(2)
        with row1 : 
            st.markdown("<h5>올바른 방법</h5>", unsafe_allow_html=True)
            st.image("/opt/ml/input/code/fastapi/app/explain_img/y.jpg", width=400, caption='정답 선지에만 "V" 표시를 해주세요')

        with row2:
            st.markdown("<h5>틀린 방법</h5>", unsafe_allow_html=True)
            st.image("/opt/ml/input/code/fastapi/app/explain_img/n1.jpg", width=400,
                    caption='정답을 "O"로 표시하지 말아주세요')
            st.image("/opt/ml/input/code/fastapi/app/explain_img/n2.jpg", width=400,
                    caption='틀린 선지에 "X" 또는 "/" 표시를 하지 말아주세요')
            st.image("/opt/ml/input/code/fastapi/app/explain_img/n3.jpg", width=400,
                    caption='정답을 선지 번호로 작성하지 말아주세요')
        


    if correct_button:
        st.write("")

        
        

    # if st.button("close"):
    #     ex_page = 1

def introduce():

    st.markdown("<h1>몇 점 일 까 ?💯</h1>", unsafe_allow_html=True)
    st.markdown("<h4>AI 채점 선생님이 당신을 대신해 채점해 드립니다</h4>", unsafe_allow_html=True)

    image_comparison(
    img1="/opt/ml/input/code/fastapi/app/explain_img/solve.jpg",
    img2="/opt/ml/input/code/fastapi/app/explain_img/check.jpg",
    label1="제출된 시험지",
    label2="Ai가 채점한 시험지",
    )
    
def main():
    global ex_page
    # if st.button("사용 설명서"):
    #      ex_page = 2
    #      page()
    st.title("몇점일까?")
    st.subheader("평가원 객관식 문제 자동채점 프로그램")

    # 스트림릿의 선택 창으로 채점할 문제의 종류를 선택하고, 정답지를 불러오는 부분입니다.
    year_choice, test_choice, type_choice = init_value()
    exam_info = year_choice + "_" + test_choice + "_" + type_choice  # ex: 2021_f_a
    response = requests.get(f"http://{backend_server}/answers/{exam_info}")
    rs = response.json()["answers"]
    if rs == "No data":
        file = st.file_uploader("정답 데이터가 없습니다 답안을 등록해주세요", type=["csv"])
        if file:
            csv_file = file.read()
            response = requests.post(
                f"http://{backend_server}/uploadfiles_name/{exam_info}",
                files={"csv_file": csv_file},
            )
            st.write("등록이 완료되었습니다.")

    uploaded_file = st.file_uploader("손으로 풀이된 시험지의 pdf파일을 업로드하세요.", type=["pdf"])

    if uploaded_file:
        # 업로드한 파일을 backend server에 보내서 모델 예측을 받는 부분입니다.
        length = 1  # TODO: uploaded_file의 길이로 수정합니다.
        files = {"file": uploaded_file.getvalue()}
        progress = stqdm(total=length)
        user_solution = requests.post(
            f"http://{backend_server}/predict/{exam_info}", files=files
        )
        progress.update(1)
        st.download_button(
            "Download Scored Image",
            data=io.BytesIO(user_solution.content).read(),
            file_name="scoring.pdf",
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    # main()
    if select == "체점하기":
        main()
    elif select == "about us":  
        introduce()
    else : 
        explain()
