import io
import os
from pathlib import Path
from pdf2image import convert_from_path
import requests
from PIL import Image
import albumentations as A 
import numpy as np 


import streamlit as st
from app.confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

root_password = 'password'


def main():
    paper = ['2021-06', '2021-09', '2021-수능', '2022-06', '2022-09', '2022-수능']
    paper_choice = st.selectbox('원하시는 시험을 선택해 주세요', paper)
    # type = ['가','나', 'a','b']
    # type_choice = st.selectbox('유형을 선택해 주세요', type)

    
    if paper_choice == '2021-06':
        type = ['가(홀)', '나(홀)', '가(짝)','나(짝)'] #11 21 12 22
        type_choice = st.selectbox('유형을 선택해 주세요', type)
        if type_choice =='가(홀)':
            read_info = requests.get("http://127.0.0.1:8001/2021-06/11")
            answer = read_info.json()['answer']
            st.write('2021-06월 모의고사 가(홀)')
        elif type_choice =='나(홀)':
            read_info = requests.get("http://127.0.0.1:8001/2021-06/21")
            answer = read_info.json()['answer']
            st.write('2021-06월 모의고사 나(홀)')
        elif type_choice =='가(짝)':
            read_info = requests.get("http://127.0.0.1:8001/2021-06/12")
            answer = read_info.json()['answer']
            st.write('2021-06월 모의고사 가(짝)')                  
        elif type_choice =='나(짝)':
            read_info = requests.get("http://127.0.0.1:8001/2021-06/22")
            answer = read_info.json()['answer']
            st.write('2021-06월 모의고사 나(짝)')    


        # st.write(answer)
    elif paper_choice == "2021-09" :
        type = ['가(홀)', '나(홀)', '가(짝)','나(짝)'] #11 21 12 22
        type_choice = st.selectbox('유형을 선택해 주세요', type)
        if type_choice =='가(홀)':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/11")
            answer = read_info.json()['answer']
            st.write('2021-09월 모의고사 가(홀)')
        elif type_choice =='나(홀)':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/21")
            answer = read_info.json()['answer']
            st.write('2021-09월 모의고사 나(홀)')
        elif type_choice =='가(짝)':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/12")
            answer = read_info.json()['answer']
            st.write('2021-09월 모의고사 가(짝)')                  
        elif type_choice =='나(짝)':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/22")
            answer = read_info.json()['answer']
            st.write('2021-09월 모의고사 나(짝)') 

    elif paper_choice == '2021-수능' :
        type = ['가(홀)', '나(홀)', '가(짝)','나(짝)'] #11 21 12 22
        type_choice = st.selectbox('유형을 선택해 주세요', type)        
        if type_choice =='가(홀)':
            read_info = requests.get("http://127.0.0.1:8001/2021-f/11")
            answer = read_info.json()['answer']
            st.write(answer)
            st.write('2021-수능 가(홀)')
        elif type_choice =='나(홀)':
            read_info = requests.get("http://127.0.0.1:8001/2021-f/21")
            answer = read_info.json()['answer']
            st.write('2021-수능 나(홀)')
        elif type_choice =='가(짝)':
            read_info = requests.get("http://127.0.0.1:8001/2021-f/12")
            answer = read_info.json()['answer']
            st.write('2021-수능 가(짝)')                  
        elif type_choice =='나(짝)':
            read_info = requests.get("http://127.0.0.1:8001/2021-f/22")
            answer = read_info.json()['answer']
            st.write('2021-수능 나(짝)') 
        exam_info = "2021_f_a"
        


    elif paper_choice == paper[3] :
        st.write('2022-06월 모의고사')
        type = ["확률과 통계", "미적분", "기하"]
        type_choice = st.selectbox('유형을 선택해 주세요', type)
        if type_choice == '확률과 통계':
            read_info = requests.get("http://127.0.0.1:8001/2022-06/probability")
            answer = read_info.json()['answer']
        elif type_choice == '미적분':
            read_info = requests.get("http://127.0.0.1:8001/2022-06/calculus")
            answer = read_info.json()['answer']    
        elif type_choice == '기하':
            read_info = requests.get("http://127.0.0.1:8001/2022-06/geometry")
            answer = read_info.json()['answer']   
        st.write('2022-06')     

    elif paper_choice == paper[4] :
        st.write('2022-09월 모의고사')  
        type = ["확률과 통계", "미적분", "기하"]
        type_choice = st.selectbox('유형을 선택해 주세요', type)
        if type_choice == '확률과 통계':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/probability")
            answer = read_info.json()['answer']
        elif type_choice == '미적분':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/calculus")
            answer = read_info.json()['answer']    
        elif type_choice == '기하':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/geometry")
            answer = read_info.json()['answer']   
        st.write('2022-수능')     

    elif paper_choice == '2022-수능':
        type = ["확률과 통계", "미적분", "기하"]
        type_choice = st.selectbox('유형을 선택해 주세요', type)
        if type_choice == '확률과 통계':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/probability")
            answer = read_info.json()['answer']
        elif type_choice == '미적분':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/calculus")
            answer = read_info.json()['answer']    
        elif type_choice == '기하':
            read_info = requests.get("http://127.0.0.1:8001/2021-09/geometry")
            answer = read_info.json()['answer']   
        st.write('2022-수능')     

    st.title("몇점이죠?")
    uploaded_file = st.file_uploader("Choose an image", type=["pdf"])

    if uploaded_file:
        save_path = Path('/opt/ml/input/code/fastapi/app/tmp', uploaded_file.name)
        with open(save_path, mode="wb") as w:
            w.write(uploaded_file.getvalue())
        images = convert_from_path(save_path)
        for idx, image in enumerate(images):
            images[idx] = A.resize(np.array(image), 3309, 2339)

        imgs_path = []
        for idx, image in enumerate(images):
            pil_image = Image.fromarray(image)
            save_path = (
                "/opt/ml/input/code/fastapi/app/tmp/"
                + str(idx + 1)
                + ".jpg"
            )
            imgs_path.append(save_path)
            pil_image.save(save_path, "JPEG")

        st.write(exam_info)
        st.write("complete")


        st.write("Classifying...")

        # 기존 stremalit 코드
        # _, y_hat = get_prediction(model, image_bytes)
        # label = config['classes'][y_hat.item()]
        files = [
            ('files', (uploaded_file.name, image_bytes,
                       uploaded_file.type))
        ]
        response = requests.post("http://localhost:8001/order", files=files)
        label = response.json()["products"][0]["result"]
        st.write(f'label is {label}')
        

@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')
