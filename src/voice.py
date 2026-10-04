import os
import streamlit as st
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()
client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

audio = st.audio_input("Say 'help me!'")

if audio:
    response = client.speech_to_text.convert(
        model_id="scribe_v2",  # use whatever model ID the ElevenLabs docs currently list
        file=audio,
    )
    text = response.text
    st.write(f"Heard: {text}")

    if "help" in text.lower():
        st.success("Distress call detected. Scanning footage...")
        # start your detection on the uploaded video here