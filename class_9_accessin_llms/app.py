import os
import io
import wave
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
    page_title="Free Multimodal AI Playground",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. LOAD ENVIRONMENT VARIABLES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

env_path = PROJECT_ROOT / ".env"

load_dotenv(env_path, override=True)


# ---------------------------------------------------------
# Try Streamlit Cloud secrets first
# Then local .env
# ---------------------------------------------------------

def get_secret(name):

    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


GEMINI_API_KEY = get_secret("GOOGLE_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")


# =========================================================
# 3. API CLIENTS
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
# 4. MODEL CONFIGURATION
# =========================================================

# Text
GEMINI_TEXT_MODEL = "gemini-3.5-flash-lite"

# Vision
GEMINI_VISION_MODEL = "gemini-3.6-flash"

# TTS
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"

# Groq chat
GROQ_TEXT_MODEL = "openai/gpt-oss-20b"

# Speech-to-text
GROQ_STT_MODEL = "whisper-large-v3-turbo"


# =========================================================
# 5. SESSION STATE
# =========================================================

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []


if "groq_messages" not in st.session_state:
    st.session_state.groq_messages = []


if "previous_task" not in st.session_state:
    st.session_state.previous_task = None


if "previous_provider" not in st.session_state:
    st.session_state.previous_provider = None


# =========================================================
# 6. CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(30, 90, 200, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(130, 50, 210, 0.12),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #07101f,
                #0b1020 50%,
                #131025
            );
    }


    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }


    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(12, 24, 48, 0.98),
                rgba(7, 14, 29, 0.98)
            );

        border-right:
            1px solid rgba(120, 140, 255, 0.15);
    }


    .title-text {

        font-size: 3rem;
        font-weight: 800;

        background:
            linear-gradient(
                90deg,
                #ffb13b,
                #ff6aa2,
                #8d73ff
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        margin-bottom: 0;
    }


    .subtitle-text {
        color: #9da8c5;
        font-size: 1rem;
        margin-bottom: 25px;
    }


    [data-testid="stChatMessage"] {

        border-radius: 18px;

        padding: 12px 16px;

        margin-bottom: 12px;

        background:
            linear-gradient(
                135deg,
                rgba(20, 30, 54, 0.94),
                rgba(20, 20, 42, 0.94)
            );

        border:
            1px solid rgba(130, 145, 255, 0.14);

        box-shadow:
            0 8px 25px rgba(0,0,0,0.18);
    }


    [data-testid="stChatInput"] {

        border-radius: 18px !important;

        border:
            1px solid rgba(110, 100, 255, 0.45) !important;

        background:
            rgba(18, 22, 40, 0.96) !important;

        box-shadow:
            0 0 22px rgba(100, 70, 255, 0.08);
    }


    div.stButton > button {

        border-radius: 14px;

        border:
            1px solid rgba(120, 130, 255, 0.20);

        background:
            linear-gradient(
                135deg,
                rgba(23, 34, 62, 0.95),
                rgba(28, 25, 52, 0.95)
            );

        color: white;

        transition: all 0.2s ease;
    }


    div.stButton > button:hover {

        transform: translateY(-1px);

        border-color:
            rgba(120, 110, 255, 0.55);

        box-shadow:
            0 8px 22px rgba(90, 70, 200, 0.16);
    }


    .api-good {

        padding: 12px;

        border-radius: 10px;

        background:
            rgba(20, 130, 90, 0.26);

        border:
            1px solid rgba(60, 220, 150, 0.15);

        color: #44f5a6;

        margin-bottom: 15px;
    }


    .model-card {

        padding: 12px;

        margin-bottom: 12px;

        border-radius: 9px;

        background:
            rgba(255,255,255,0.05);

        border:
            1px solid rgba(255,255,255,0.04);

        font-family: monospace;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## API Status")


    if GEMINI_API_KEY:

        st.markdown(
            '<div class="api-good">'
            'Google Gemini API connected'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.error(
            "Gemini API key missing"
        )


    if GROQ_API_KEY:

        st.markdown(
            '<div class="api-good">'
            'Groq API connected'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.error(
            "Groq API key missing"
        )


    st.markdown("---")


    st.markdown(
        "### Available models"
    )


    st.markdown(
        "**Gemini text:**"
    )

    st.markdown(
        f'<div class="model-card">'
        f'{GEMINI_TEXT_MODEL}'
        f'</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        "**Gemini vision:**"
    )

    st.markdown(
        f'<div class="model-card">'
        f'{GEMINI_VISION_MODEL}'
        f'</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        "**Groq text:**"
    )

    st.markdown(
        f'<div class="model-card">'
        f'{GROQ_TEXT_MODEL}'
        f'</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        "**Groq speech-to-text:**"
    )

    st.markdown(
        f'<div class="model-card">'
        f'{GROQ_STT_MODEL}'
        f'</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        "**Gemini TTS:**"
    )

    st.markdown(
        f'<div class="model-card">'
        f'{GEMINI_TTS_MODEL}'
        f'</div>',
        unsafe_allow_html=True
    )


    st.markdown("---")


    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):

        st.session_state.gemini_messages = []
        st.session_state.groq_messages = []

        st.rerun()


# =========================================================
# 8. PAGE TITLE
# =========================================================

st.markdown(
    '<p class="title-text">'
    '✨ Free Multimodal AI Playground'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle-text">'
    'Gemini + Groq | Text, Image and Audio'
    '</p>',
    unsafe_allow_html=True
)


# =========================================================
# 9. TASK SELECTION
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
# 10. TEXT → TEXT
# =========================================================

if task == "Text → Text":

    provider = st.selectbox(
        "Provider",
        [
            "Groq",
            "Gemini"
        ]
    )


    # -----------------------------------------------------
    # Detect provider switch
    # -----------------------------------------------------

    if (
        st.session_state.previous_provider is not None
        and
        st.session_state.previous_provider != provider
    ):

        # Keep histories separate
        pass


    st.session_state.previous_provider = provider


    # -----------------------------------------------------
    # Choose correct history
    # -----------------------------------------------------

    if provider == "Gemini":

        messages = st.session_state.gemini_messages

    else:

        messages = st.session_state.groq_messages


    # -----------------------------------------------------
    # Small controls
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [6, 1]
    )


    with col1:

        st.caption(
            f"Conversation using {provider}"
        )


    with col2:

        if st.button(
            "Clear",
            key=f"clear_{provider}"
        ):

            if provider == "Gemini":
                st.session_state.gemini_messages = []

            else:
                st.session_state.groq_messages = []

            st.rerun()


    # -----------------------------------------------------
    # Display conversation history
    # -----------------------------------------------------

    for message in messages:

        avatar = (
            "🧑"
            if message["role"] == "user"
            else "✨"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # Chat input
    # -----------------------------------------------------

    prompt = st.chat_input(
        f"Message {provider}..."
    )


    if prompt:

        # ---------------------------------------------
        # Save user message
        # ---------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )


        with st.chat_message(
            "user",
            avatar="🧑"
        ):

            st.markdown(prompt)


        # =============================================
        # GEMINI CHAT
        # =============================================

        if provider == "Gemini":

            if not gemini_client:

                st.error(
                    "Gemini API is not configured."
                )

            else:

                with st.chat_message(
                    "assistant",
                    avatar="✨"
                ):

                    placeholder = st.empty()

                    full_response = ""

                    try:

                        conversation = []

                        for msg in messages:

                            role = (
                                "user"
                                if msg["role"] == "user"
                                else "model"
                            )

                            conversation.append(
                                {
                                    "role": role,
                                    "parts": [
                                        {
                                            "text":
                                            msg["content"]
                                        }
                                    ]
                                }
                            )


                        stream = (
                            gemini_client
                            .models
                            .generate_content_stream(
                                model=
                                GEMINI_TEXT_MODEL,

                                contents=
                                conversation
                            )
                        )


                        for chunk in stream:

                            if chunk.text:

                                full_response += (
                                    chunk.text
                                )

                                placeholder.markdown(
                                    full_response
                                    + "▌"
                                )


                        placeholder.markdown(
                            full_response
                        )


                        messages.append(
                            {
                                "role":
                                "assistant",

                                "content":
                                full_response
                            }
                        )


                    except Exception as e:

                        placeholder.error(
                            str(e)
                        )


        # =============================================
        # GROQ CHAT
        # =============================================

        else:

            if not groq_client:

                st.error(
                    "Groq API is not configured."
                )

            else:

                with st.chat_message(
                    "assistant",
                    avatar="✨"
                ):

                    placeholder = st.empty()

                    full_response = ""

                    try:

                        groq_messages = [
                            {
                                "role":
                                msg["role"],

                                "content":
                                msg["content"]
                            }

                            for msg in messages
                        ]


                        stream = (
                            groq_client
                            .chat
                            .completions
                            .create(
                                model=
                                GROQ_TEXT_MODEL,

                                messages=
                                groq_messages,

                                stream=True
                            )
                        )


                        for chunk in stream:

                            content = (
                                chunk
                                .choices[0]
                                .delta
                                .content
                            )


                            if content:

                                full_response += (
                                    content
                                )

                                placeholder.markdown(
                                    full_response
                                    + "▌"
                                )


                        placeholder.markdown(
                            full_response
                        )


                        messages.append(
                            {
                                "role":
                                "assistant",

                                "content":
                                full_response
                            }
                        )


                    except Exception as e:

                        placeholder.error(
                            str(e)
                        )


# =========================================================
# 11. IMAGE → TEXT
# =========================================================

elif task == "Image → Text":

    st.subheader(
        "🖼️ Image → Text"
    )


    if not gemini_client:

        st.error(
            "Gemini API is not configured."
        )

        st.stop()


    uploaded_image = st.file_uploader(
        "Upload an image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ]
    )


    image_prompt = st.text_area(
        "What do you want to know about the image?",
        value="Describe this image in detail."
    )


    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded image",
            use_container_width=True
        )


        if st.button(
            "Analyze image",
            use_container_width=True
        ):

            try:

                image_bytes = (
                    uploaded_image.getvalue()
                )


                mime_type = (
                    uploaded_image.type
                )


                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model=
                        GEMINI_VISION_MODEL,

                        contents=[
                            image_prompt,

                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type
                            )
                        ]
                    )
                )


                st.markdown(
                    "### Response"
                )

                st.markdown(
                    response.text
                )


            except Exception as e:

                st.error(
                    str(e)
                )


# =========================================================
# 12. AUDIO → TEXT
# =========================================================

elif task == "Audio → Text":

    st.subheader(
        "🎙️ Audio → Text"
    )


    if not groq_client:

        st.error(
            "Groq API is not configured."
        )

        st.stop()


    audio_file = st.file_uploader(
        "Upload an audio file",
        type=[
            "wav",
            "mp3",
            "m4a",
            "ogg",
            "webm",
            "flac"
        ]
    )


    if audio_file:

        st.audio(
            audio_file
        )


        if st.button(
            "Transcribe audio",
            use_container_width=True
        ):

            try:

                audio_bytes = (
                    audio_file.getvalue()
                )


                transcription = (
                    groq_client
                    .audio
                    .transcriptions
                    .create(
                        file=(
                            audio_file.name,
                            audio_bytes
                        ),

                        model=
                        GROQ_STT_MODEL
                    )
                )


                st.markdown(
                    "### Transcription"
                )

                st.write(
                    transcription.text
                )


            except Exception as e:

                st.error(
                    str(e)
                )


# =========================================================
# 13. TEXT → AUDIO
# =========================================================

elif task == "Text → Audio":

    st.subheader(
        "🔊 Text → Audio"
    )


    if not gemini_client:

        st.error(
            "Gemini API is not configured."
        )

        st.stop()


    tts_text = st.text_area(
        "Enter text",
        placeholder=(
            "Type something you want "
            "Gemini to speak..."
        )
    )


    voice = st.selectbox(
        "Voice",
        [
            "Kore",
            "Leda"
        ]
    )


    if st.button(
        "Generate speech",
        use_container_width=True
    ):

        if not tts_text.strip():

            st.warning(
                "Enter some text first."
            )

        else:

            try:

                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model=
                        GEMINI_TTS_MODEL,

                        contents=
                        tts_text,

                        config=
                        types.GenerateContentConfig(
                            response_modalities=[
                                "AUDIO"
                            ],

                            speech_config=
                            types.SpeechConfig(

                                voice_config=
                                types.VoiceConfig(

                                    prebuilt_voice_config=
                                    types
                                    .PrebuiltVoiceConfig(
                                        voice_name=
                                        voice
                                    )
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


                # -------------------------------------
                # Convert raw PCM to WAV
                # -------------------------------------

                wav_buffer = io.BytesIO()


                with wave.open(
                    wav_buffer,
                    "wb"
                ) as wav_file:

                    wav_file.setnchannels(1)

                    wav_file.setsampwidth(2)

                    wav_file.setframerate(
                        24000
                    )

                    wav_file.writeframes(
                        audio_data
                    )


                wav_buffer.seek(0)


                st.audio(
                    wav_buffer,
                    format="audio/wav"
                )


                st.download_button(
                    "Download audio",
                    data=wav_buffer,
                    file_name=
                    "generated_speech.wav",
                    mime="audio/wav"
                )


            except Exception as e:

                st.error(
                    str(e)
                )


# =========================================================
# 14. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Free API access is subject to provider quotas "
    "and rate limits."
)