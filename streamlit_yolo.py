import cv2
import os
from ultralytics import YOLO
import time
import datetime
import streamlit as st


st.title('OpenCV + Yolo8 + Streamlit')

st.sidebar.header('Настройки')
model_name = st.sidebar.selectbox('Выберите модель:', ['yolov8n', 'yoloilin.pt'])

count_conf = st.sidebar.slider('Мин точность: ', 0.25, 0.9, 0.5,0.05)
start_button = st.sidebar.button('Запустить')

model = YOLO(model_name)

print_image = st.image([])
if start_button:
    st.success('Модель загружена')
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        ret, frame =  cap.read()

        if not ret:
            st.error('Камера не работает')
            break
        result = model(frame, stream=True, conf=count_conf)

        for i in result:
            for n in i.boxes:
                cls = int(n.cls[0])
                label = model.names[cls]
                conf = round(float(n.conf[0]), 2)
                x, y, w, h = map(int, n.xyxy[0])
                cv2.rectangle(frame, (x,y), (w,h),  (0,255,0), 2)
                cv2.putText(frame, f'{label}, {conf*100}%', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)


            end_time = time.time()
            fps = 1 /  (end_time - start_time)
            cv2.putText(frame, f'FPS:{fps}', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print_image.image(rgb_frame)
    cap.release()
    st.info('Видео остановлено')
else:
    st.info('Нажмите на кнопку чтобы запустить YOLO')