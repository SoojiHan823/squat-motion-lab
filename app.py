
import streamlit as st
import sys

st.title("Squat Motion Lab - 환경 점검")

st.write("Python:", sys.version)

try:
    import cv2
    st.success("OpenCV 정상 실행")
    st.write("OpenCV:", cv2.__version__)
except Exception as e:
    st.error(f"OpenCV 오류: {type(e).__name__}")
    st.code(str(e))
    st.exception(e)

try:
    import mediapipe as mp
    st.success("MediaPipe 정상 실행")
    st.write("MediaPipe:", mp.__version__)
except Exception as e:
    st.error(f"MediaPipe 오류: {type(e).__name__}")
    st.code(str(e))
    st.exception(e)
