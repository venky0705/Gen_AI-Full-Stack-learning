import os
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
# 2. LOAD API KEYS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Local .env
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)


def get_secret(name):
    """
    First try environment variables.
    If not found, try Streamlit Cloud secrets.
    """

    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets[name]
    except Exception:
        return None


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")


# =========================================================
# 3. CLIENTS
# =========================================================

gemini_client = None
groq_client = None

if GOOGLE_API_KEY:
    gemini_client = genai.Client(
        api_key=GOOGLE_API_KEY
    )

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY
    )


# =========================================================
# 4. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 18% 8%,
                rgba(50, 100, 255, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 82% 12%,
                rgba(170, 70, 255, 0.12),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #070b14,
                #0b1020,
                #11101f
            );
    }

    .main .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(16, 24, 45, 0.98),
                rgba(8, 13, 26, 0.98)
            );

        border-right:
            1px solid rgba(130, 150, 255, 0.15);
    }

    div.stButton > button {
        min-height: 48px;
        border-radius: 14px;

        border:
            1px solid rgba(120, 145, 255, 0.20);

        background:
            linear-gradient(
                135deg,
                rgba(20, 30, 55, 0.92),
                rgba(22, 22, 45, 0.92)
            );

        color: white;

        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);

        border-color:
            rgba(130, 120, 255, 0.60);

        box-shadow:
            0 8px 24px rgba(80, 70, 200, 0.22);
    }

    [data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def get_gemini_text(response):
    """
    Safely extract text from Gemini response.
    """

    if hasattr(response, "text") and response.text:
        return response.text

    return str(response)


def gemini_text(prompt):
    """
    Text -> Text using Gemini
    """

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return get_gemini_text(response)


def gemini_image_to_text(uploaded_file, prompt):
    """
    Image -> Text using Gemini vision
    """

    image_bytes = uploaded_file.getvalue()

    mime_type = (
        uploaded_file.type
        or "image/jpeg"
    )

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
    Text -> Text using Groq
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
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        temp_path = temp_file.name

    try:

        with open(
            temp_path,
            "rb"
        ) as audio_file:

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
    Save Gemini PCM audio as WAV.
    """

    import wave

    with wave.open(
        filename,
        "wb"
    ) as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)

    return filename


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("API Status")

    if GOOGLE_API_KEY:
        st.success("Google Gemini API connected")
    else:
        st.error("Google API key missing")

    if GROQ_API_KEY:
        st.success("Groq API connected")
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
# 7. HEADER
# =========================================================

st.title("✨ Free Multimodal AI Playground")

st.caption(
    "Gemini + Groq | Text, Image and Audio"
)


# =========================================================
# 8. TASK SELECTION
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

            st.warning(
                "Please enter a prompt."
            )

        else:

            try:

                with st.spinner(
                    "Generating..."
                ):

                    if provider == "Groq":

                        if not groq_client:
                            st.error(
                                "GROQ_API_KEY not configured."
                            )
                            st.stop()

                        result = groq_text(
                            prompt
                        )

                    else:

                        if not gemini_client:
                            st.error(
                                "GOOGLE_API_KEY not configured."
                            )
                            st.stop()

                        result = gemini_text(
                            prompt
                        )

                st.subheader(
                    "Response"
                )

                st.markdown(
                    result
                )

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
                "GOOGLE_API_KEY not configured."
            )

        elif image_file is None:

            st.warning(
                "Please upload an image first."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing image..."
                ):

                    result = (
                        gemini_image_to_text(
                            image_file,
                            prompt
                        )
                    )

                st.subheader(
                    "Image description"
                )

                st.markdown(
                    result
                )

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

        st.audio(
            audio_file
        )

    if st.button(
        "Transcribe",
        use_container_width=True
    ):

        if not groq_client:

            st.error(
                "GROQ_API_KEY not configured."
            )

        elif audio_file is None:

            st.warning(
                "Please upload an audio file first."
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

                st.subheader(
                    "Transcription"
                )

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
                "GOOGLE_API_KEY not configured."
            )

        elif not text.strip():

            st.warning(
                "Please enter some text."
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

                st.subheader(
                    "Generated speech"
                )

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
    "Free API access is subject to "
    "provider quotas and rate limits."
)