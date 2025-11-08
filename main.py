from time import strftime
import cv2
import numpy as np
import streamlit as st
import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


st.set_page_config(page_title='VideoFilter', layout='centered')

st.title('Camera Filteres')

filter_option = st.radio(
    'Select a filter',
    ('Normal', 'Black/white', 'Blur', 'Contr'),
    horizontal=True
)

class VideoCamera(VideoTransformerBase):
    def __ini__(self):
        self.filter_option = 'Normal'

    def transform(self, frame):
        img = frame.to_ndarray(format='bgr24')

        text = datetime.datetime.now().strftime('%d-%m-%y %M:%H:%S')
        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0  ), 2, cv2.LINE_AA)

        if self.filter_option == 'Black/white':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        elif self.filter_option == 'Blur':
            img = cv2.GaussianBlur(img, (35, 35), 0)

        elif self.filter_option == 'Contr':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 100)
            img = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

        return img

webrtc_ctx = webrtc_streamer(
    key='video',
    video_processor_factory=VideoCamera,
    media_stream_constraints={'video' : True, 'audio' : False}
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.filter_option = filter_option
