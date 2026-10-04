import os
import tempfile
import cv2
import streamlit as st
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from ultralytics import YOLO

st.set_page_config(page_title="Humanitarian Drone Detection System", layout="wide")

load_dotenv()
client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

# Path to best.pt, relative to this file. Adjust if it lives somewhere else.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")


@st.cache_resource  # load the model once, not on every rerun
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


st.title("Humanitarian Drone Detection System")
st.caption("Search and rescue support for crisis response.")


def run_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_slot = st.empty()   # one spot on the page that updates each frame
    count_slot = st.empty()
    frame_num = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_num += 1
        if frame_num % 5 != 0:  # process every 5th frame for speed
            continue

        # Shrink huge 4K frames so display and inference stay fast
        h, w = frame.shape[:2]
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))

        # Larger imgsz helps with tiny humans seen from above
        result = model(frame, imgsz=1280, conf=0.25, verbose=False)[0]
        frame_slot.image(result.plot(), channels="BGR")
        count_slot.metric("Detections in current frame", len(result.boxes))

    cap.release()
    st.success("Scan complete.")


# 1. Upload the footage
video = st.file_uploader("Upload drone footage", type=["mp4", "mov"])
video_path = None

if video:
    st.video(video)
    # OpenCV needs a real file path, so save a temp copy
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.getvalue())
        video_path = tmp.name

# 2. Voice trigger
audio = st.audio_input("Say 'help me!'")

if audio:
    response = client.speech_to_text.convert(
        model_id="scribe_v2",  # use whatever model ID the docs currently list
        file=audio,
    )
    text = response.text
    st.write(f"Heard: {text}")

    if "help" in text.lower():
        if video_path:
            st.success("Distress call detected. Scanning footage...")
            run_detection(video_path)
        else:
            st.warning("Distress call heard, but no footage uploaded yet.")