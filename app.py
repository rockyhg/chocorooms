import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from models.detect import detect_kinotake


class VideoProcessor:
    def __init__(self) -> None:
        self.state = False
        self.threshold = 0.6
        self.disp_score = False
        self.disp_counter = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.state:
            img = detect_kinotake(
                img,
                confidence_threshold=self.threshold,
                disp_score=self.disp_score,
                disp_counter=self.disp_counter,
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            pass

        img = av.VideoFrame.from_ndarray(img, format="bgr24")
        return img


def main():
    st.title("Kinoko Takenoko Detection")
    st.caption("「きのこの山」と「たけのこの里」を検出します")

    ctx = webrtc_streamer(
        key="chocorooms",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    if ctx.video_processor:
        ctx.video_processor.state = st.checkbox("DETECTION")
        with st.sidebar:
            ctx.video_processor.disp_score = st.checkbox("Score", value=False)
            ctx.video_processor.disp_counter = st.checkbox("Counter", value=True)
            ctx.video_processor.threshold = st.slider(
                "Threshold", min_value=0.0, max_value=1.0, step=0.05, value=0.6
            )


if __name__ == "__main__":
    main()
