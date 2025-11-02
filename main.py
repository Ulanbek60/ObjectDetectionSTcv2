import streamlit as st
import cv2
import datetime
import tempfile

st.set_page_config(page_title="Webcam Stream", layout="centered")

st.title("Camera with filteres")

# Выбор фильтра
filter_option = st.selectbox(
    "Выбери фильтр:",
    ("normal", "gray", "blur", "lines")
)

start_button = st.button("▶️ Начать запись")

frame_placeholder = st.empty()

if start_button:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Камера не найдена.")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmpfile.name, fourcc, frame_fps, (frame_width, frame_height))

    stop_button = st.button("⏹️ Остановить")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Кадр не найден.")
            break

        text = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        cv2.putText(frame, text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, cv2.LINE_AA)

        if filter_option == "gray":
            filter_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filter_frame = cv2.cvtColor(filter_frame, cv2.COLOR_GRAY2BGR)
        elif filter_option == "blur":
            filter_frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif filter_option == "lines":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            filter_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            filter_frame = frame

        out.write(filter_frame)

        frame_placeholder.image(
            cv2.cvtColor(filter_frame, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

    cap.release()
    out.release()
    st.success("Запись завершена")
    st.video(tmpfile.name)
