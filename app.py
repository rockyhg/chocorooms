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
        self.weights = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.state:
            img = detect_kinotake(
                img,
                weights_file=self.weights,
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
            ctx.video_processor.weights = st.selectbox(
                "Select Weights:",
                ["kinotake_ssd_v1.pth", "kinotake_ssd_v2.pth", "kinotake_ssd_v3.pth"],
                index=2,
            )
            ctx.video_processor.disp_score = st.checkbox("Score", value=False)
            ctx.video_processor.disp_counter = st.checkbox("Counter", value=True)
            ctx.video_processor.threshold = st.slider(
                "Threshold", min_value=0.0, max_value=1.0, step=0.05, value=0.6
            )

    with st.expander("開発ストーリー"):
        st.markdown("**データ収集**")
        st.markdown("- iPhone + ミニ三脚で撮影")
        st.markdown("- 画像サイズ: 1200 x 1200")
        st.markdown("- きのこの山6個、たけのこの里6個を1セットとし、1セットあたり10枚撮影")
        st.markdown("- きのこたけのこそれぞれ1箱分のサンプルを使用し、合計100枚のデータを用意")
        st.markdown("")
        st.image("./docs/taking_photo.jpg", width=300)
        st.image("./docs/kinoko-takenoko.jpg", width=300)
        st.markdown("")
        st.markdown("**前処理（アノテーション）**")
        st.markdown("- labelImgを使用")
        st.markdown("- すべてのオブジェクトをボックスで囲む")
        st.markdown("")
        st.image("./docs/annotation.png", width=480)
        st.markdown("")
        st.markdown("**モデル作成**")
        st.markdown("- SSD (Single Shot MultiBox Detector) を使用")
        st.markdown("- 参考情報[2]のコードを転用")
        st.markdown("")
        st.markdown("**学習**")
        st.markdown("- データ拡張（ランダムに拡大・切り出し・反転）")
        st.markdown("- 500エポックを学習。Colab（GPU使用）で2時間程度")
        st.markdown("")
        st.markdown("**Streamlit**")
        st.markdown("- Webカメラの映像をリアルタイム表示")
        st.markdown("- 使用ライブラリ: [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)")
        st.markdown("")
        st.markdown("**参考にした情報**")
        st.markdown("- [1] キカガク 画像処理特化コース")
        st.markdown("- [2] チーム・カルポ「[物体検出とGAN、オートエンコーダー、画像処理入門](https://www.amazon.co.jp/gp/product/B09MHLC3F8/)」")
        st.markdown("")
        st.image("https://m.media-amazon.com/images/I/51jOT49zsAL._SX386_BO1,204,203,200_.jpg", width=128)
        st.markdown("")
        st.markdown("**リポジトリ**")
        st.markdown("- [GitHub - rockyhg/chocorooms](https://github.com/rockyhg/chocorooms)")


if __name__ == "__main__":
    main()
