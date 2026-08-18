import os
import base64
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from google import genai
from google.genai import types

from groq import Groq


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Free Multimodal AI",
    page_icon="✨",
    layout="wide"
)


# =========================================================
# 2. LOAD ENVIRONMENT VARIABLES
# =========================================================

# If app.py is in a subfolder and .env is one level above:
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=True)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================================================
# 3. CLIENTS
# =========================================================

gemini_client = None
groq_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY
    )


# =========================================================
# 4. CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(60,100,255,0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at top right,
                rgba(160,70,255,0.12),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #070b14,
                #0b1020,
                #11101f
            );
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    div.stButton > button {
        border-radius: 14px;
        min-height: 46px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 5. HELPERS
# =========================================================

def get_gemini_text(response):
    """
    Handles Gemini responses that may return
    plain text or structured content.
    """

    if hasattr(response, "text") and response.text:
        return response.text

    return str(response)


def gemini_text(prompt):
    """
    Text -> Text
    """

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return get_gemini_text(response)


def gemini_image_to_text(uploaded_file, prompt):
    """
    Image -> Text
    """

    image_bytes = uploaded_file.getvalue()

    mime_type = uploaded_file.type or "image/jpeg"

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            prompt or "Describe this image in detail.",
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
        ]
    )

    return get_gemini_text(response)


def groq_text(prompt):
    """
    Text -> Text
    """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def groq_audio_to_text(uploaded_file):
    """
    Audio -> Text using Groq Whisper
    """

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(uploaded_file.getvalue())

        temp_path = temp.name

    try:

        with open(temp_path, "rb") as audio_file:

            transcription = (
                groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
            )

        return transcription

    finally:

        try:
            os.remove(temp_path)
        except Exception:
            pass


def gemini_text_to_speech(text):
    """
    Text -> Audio using Gemini TTS

    Returns raw PCM audio bytes.
    """

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-tts-preview",

        contents=(
            "Speak naturally and clearly: "
            + text
        ),

        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],

            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=
                    types.PrebuiltVoiceConfig(
                        voice_name="Kore"
                    )
                )
            )
        )
    )

    audio_data = (
        response
        .candidates[0]
        .content
        .parts[0]
        .inline_data
        .data
    )

    return audio_data


def save_pcm_as_wav(
    audio_data,
    filename="generated_audio.wav"
):
    """
    Gemini TTS returns PCM audio.
    Convert it to WAV.
    """

    import wave

    with wave.open(filename, "wb") as wf:

        wf.setnchannels(1)

        wf.setsampwidth(2)

        wf.setframerate(24000)

        wf.writeframes(audio_data)

    return filename


# =========================================================
# 6. HEADER
# =========================================================

st.title("✨ Free Multimodal AI Playground")

st.caption(
    "Gemini + Groq | Text, Image and Audio"
)


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("API Status")

    if GEMINI_API_KEY:
        st.success("Gemini connected")
    else:
        st.error("Gemini API key missing")

    if GROQ_API_KEY:
        st.success("Groq connected")
    else:
        st.error("Groq API key missing")

    st.markdown("---")

    st.subheader("Available models")

    st.write("Gemini text:")
    st.code("gemini-3.5-flash-lite")

    st.write("Gemini vision:")
    st.code("gemini-3.6-flash")

    st.write("Groq text:")
    st.code("openai/gpt-oss-20b")

    st.write("Groq speech-to-text:")
    st.code("whisper-large-v3-turbo")

    st.write("Gemini TTS:")
    st.code("gemini-3.1-flash-tts-preview")


# =========================================================
# 8. MODE SELECTION
# =========================================================

task = st.selectbox(
    "Choose task",
    [
        "Text → Text",
        "Image → Text",
        "Audio → Text",
        "Text → Audio"
    ]
)


# =========================================================
# 9. TEXT -> TEXT
# =========================================================

if task == "Text → Text":

    provider = st.selectbox(
        "Provider",
        [
            "Groq",
            "Gemini"
        ]
    )

    prompt = st.text_area(
        "Enter your prompt",
        height=160,
        placeholder="Ask anything..."
    )

    if st.button(
        "Generate",
        use_container_width=True
    ):

        if not prompt.strip():

            st.warning("Enter a prompt.")

        else:

            try:

                with st.spinner("Generating..."):

                    if provider == "Groq":

                        if not groq_client:
                            st.error(
                                "GROQ_API_KEY not configured."
                            )
                            st.stop()

                        result = groq_text(prompt)

                    else:

                        if not gemini_client:
                            st.error(
                                "GEMINI_API_KEY not configured."
                            )
                            st.stop()

                        result = gemini_text(prompt)

                st.subheader("Response")

                st.markdown(result)

            except Exception as e:

                st.error(
                    f"Generation failed:\n\n{e}"
                )


# =========================================================
# 10. IMAGE -> TEXT
# =========================================================

elif task == "Image → Text":

    st.info(
        "Uses Gemini vision."
    )

    image_file = st.file_uploader(
        "Upload image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ]
    )

    prompt = st.text_input(
        "Prompt",
        value="Describe this image in detail."
    )

    if image_file:

        st.image(
            image_file,
            caption="Uploaded image",
            width=500
        )

    if st.button(
        "Analyze Image",
        use_container_width=True
    ):

        if not gemini_client:

            st.error(
                "Gemini API key missing."
            )

        elif image_file is None:

            st.warning(
                "Upload an image first."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing image..."
                ):

                    result = gemini_image_to_text(
                        image_file,
                        prompt
                    )

                st.subheader("Image description")

                st.markdown(result)

            except Exception as e:

                st.error(
                    f"Image analysis failed:\n\n{e}"
                )


# =========================================================
# 11. AUDIO -> TEXT
# =========================================================

elif task == "Audio → Text":

    st.info(
        "Uses Groq Whisper."
    )

    audio_file = st.file_uploader(
        "Upload audio",
        type=[
            "mp3",
            "wav",
            "m4a",
            "ogg"
        ]
    )

    if audio_file:

        st.audio(audio_file)

    if st.button(
        "Transcribe",
        use_container_width=True
    ):

        if not groq_client:

            st.error(
                "Groq API key missing."
            )

        elif audio_file is None:

            st.warning(
                "Upload an audio file first."
            )

        else:

            try:

                with st.spinner(
                    "Transcribing..."
                ):

                    transcript = (
                        groq_audio_to_text(
                            audio_file
                        )
                    )

                st.subheader("Transcription")

                st.markdown(
                    str(transcript)
                )

            except Exception as e:

                st.error(
                    f"Transcription failed:\n\n{e}"
                )


# =========================================================
# 12. TEXT -> AUDIO
# =========================================================

elif task == "Text → Audio":

    st.info(
        "Uses Gemini TTS."
    )

    text = st.text_area(
        "Enter text",
        height=180,
        placeholder=(
            "Enter the text you want "
            "to convert to speech..."
        )
    )

    if st.button(
        "Generate Speech",
        use_container_width=True
    ):

        if not gemini_client:

            st.error(
                "Gemini API key missing."
            )

        elif not text.strip():

            st.warning(
                "Enter some text."
            )

        else:

            try:

                with st.spinner(
                    "Generating speech..."
                ):

                    pcm_audio = (
                        gemini_text_to_speech(
                            text
                        )
                    )

                    filename = (
                        save_pcm_as_wav(
                            pcm_audio
                        )
                    )

                st.subheader("Generated speech")

                st.audio(
                    filename,
                    format="audio/wav"
                )

            except Exception as e:

                st.error(
                    f"TTS failed:\n\n{e}"
                )


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Free API access is subject to provider "
    "rate limits and quotas."
)