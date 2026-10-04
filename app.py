import os
import tempfile
import cv2
import streamlit as st
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types
from ultralytics import YOLO

st.set_page_config(page_title="Laila", layout="wide")

load_dotenv()
client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
GEMINI_MODEL = "gemini-3.8-flash"  # check AI Studio for the current Flash model ID
VOICE_ID = "PIGsltMj3gFMR34aFDI3"  # any voice ID from your ElevenLabs voice library


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

st.title("Laila")
st.caption("Search and rescue support for crisis response.")


def speak(text):
    """Turn text into speech with ElevenLabs and play it in the page."""
    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_flash_v2_5",  # fast model, good for live demos
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_stream)
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"Voice output failed: {e}")


def grab_frame(video_path):
    """Grab a frame from the middle of the video as JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (1024, int(h * 1024 / w)))
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def describe_scene(jpg_bytes):
    prompt = (
        "You are an aerial disaster-response analyst looking at a drone frame. "
        "In one or two short sentences, state which disaster conditions are "
        "visible (for example wildfire, smoke, flooding, collapsed buildings, "
        "storm damage). Only report what you can actually see. If there is no "
        "visible disaster, say so. Write it so it sounds natural when spoken aloud."
    )
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    return response.text.strip()


def run_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_slot = st.empty()
    count_slot = st.empty()
    frame_num = 0
    max_people = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_num += 1
        if frame_num % 5 != 0:
            continue

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (1280, int(h * 1280 / w)))

        result = model(frame, imgsz=1280, conf=0.25, verbose=False)[0]
        n = len(result.boxes)
        max_people = max(max_people, n)
        frame_slot.image(result.plot(), channels="BGR")
        count_slot.metric("Detections in current frame", n)

    cap.release()
    return max_people


# 1. Upload the footage
video = st.file_uploader("Upload drone footage", type=["mp4", "mov"])
video_path = None

if video:
    st.video(video)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.getvalue())
        video_path = tmp.name

# 2. Voice trigger
audio = st.audio_input("Say 'help me!'")

if audio:
    response = client.speech_to_text.convert(model_id="scribe_v2", file=audio)
    text = response.text
    st.write(f"Heard: {text}")

    if "help" in text.lower():
        if video_path:
            st.success("Distress call detected. Analyzing environment...")

            # Gemini describes the scene, then the app says it out loud
            frame_bytes = grab_frame(video_path)
            if frame_bytes:
                try:
                    description = describe_scene(frame_bytes)
                    st.info(description)
                    speak(f"Distress call received. {description}")
                except Exception as e:
                    st.warning(f"Scene analysis failed: {e}")

            # YOLO scans for people, then the app reports the count
            st.write("Scanning for survivors...")
            people = run_detection(video_path)
            summary = (
                f"Scan complete. Up to {people} people detected in a single frame."
                if people
                else "Scan complete. No people detected."
            )
            st.success(summary)
            speak(summary)
        else:
            st.warning("Distress call heard, but no footage uploaded yet.")