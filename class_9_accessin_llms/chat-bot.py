import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai


# =========================================================
# 1. LOAD .ENV
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=True)

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY not found in .env file.")
    st.stop()


# =========================================================
# 2. GEMINI CLIENT
# =========================================================

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.1-flash-lite"


# =========================================================
# 3. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Gemini Local Chat",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
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
                circle at 18% 10%,
                rgba(0, 120, 255, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 82% 18%,
                rgba(175, 70, 255, 0.14),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #070b14 0%,
                #0b1020 45%,
                #11101f 100%
            );
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(16, 24, 45, 0.97),
                rgba(8, 13, 26, 0.98)
            );

        border-right:
            1px solid rgba(130, 150, 255, 0.15);

        backdrop-filter: blur(20px);
    }

    .gemini-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 6px;

        background:
            linear-gradient(
                90deg,
                #38a5ff,
                #7477ff,
                #c35cff
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        text-shadow:
            0 0 30px rgba(100, 100, 255, 0.18);
    }

    .gemini-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #aeb6d4;
        margin-bottom: 15px;
    }

    .gradient-line {
        width: 90px;
        height: 4px;
        margin: auto;
        margin-bottom: 35px;

        border-radius: 50px;

        background:
            linear-gradient(
                90deg,
                #38a5ff,
                #8d68ff,
                #db54ff
            );

        box-shadow:
            0 0 18px rgba(130, 90, 255, 0.65);
    }

    [data-testid="stChatMessage"] {
        border-radius: 20px;
        padding: 14px 18px;
        margin-bottom: 14px;

        background:
            linear-gradient(
                135deg,
                rgba(18, 28, 50, 0.92),
                rgba(18, 20, 38, 0.92)
            );

        border:
            1px solid rgba(120, 145, 255, 0.16);

        box-shadow:
            0 8px 28px rgba(0, 0, 0, 0.20),
            inset 0 1px 0 rgba(255,255,255,0.03);

        backdrop-filter: blur(18px);
    }

    [data-testid="stChatInput"] {
        border-radius: 18px !important;

        border:
            1px solid rgba(110, 120, 255, 0.55) !important;

        background:
            linear-gradient(
                135deg,
                rgba(20, 25, 44, 0.98),
                rgba(28, 23, 45, 0.98)
            ) !important;

        box-shadow:
            0 0 28px rgba(100, 70, 255, 0.12),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    div.stButton > button {
        min-height: 52px;
        border-radius: 16px;

        border:
            1px solid rgba(120, 145, 255, 0.20);

        background:
            linear-gradient(
                135deg,
                rgba(20, 30, 55, 0.92),
                rgba(22, 22, 45, 0.92)
            );

        color: white;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            border-color 0.2s ease;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);

        border-color:
            rgba(130, 120, 255, 0.60);

        box-shadow:
            0 10px 30px rgba(80, 70, 200, 0.22),
            0 0 18px rgba(110, 80, 255, 0.18);
    }

    .status-card {
        padding: 14px;
        border-radius: 16px;

        background:
            linear-gradient(
                135deg,
                rgba(0, 120, 90, 0.20),
                rgba(0, 70, 65, 0.25)
            );

        border:
            1px solid rgba(50, 220, 170, 0.25);

        color: #72f7c3;

        font-size: 0.95rem;
    }

    .power-card {
        margin-top: 25px;
        padding: 22px;

        text-align: center;

        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                rgba(90, 40, 150, 0.20),
                rgba(20, 60, 130, 0.18)
            );

        border:
            1px solid rgba(140, 100, 255, 0.35);

        box-shadow:
            0 0 30px rgba(110, 70, 255, 0.10);
    }

    .power-title {
        font-size: 1.1rem;
        font-weight: 700;

        background:
            linear-gradient(
                90deg,
                #8f7cff,
                #d36cff
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 5. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ✨ Gemini Chat")
    st.caption("Local AI chatbot")

    st.markdown("---")

    st.markdown("### Model")
    st.code(MODEL_NAME)

    st.markdown("---")

    st.markdown("### Conversation")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")

    st.write(
        f"Messages: {len(st.session_state.messages)}"
    )

    st.markdown(
        '<div class="status-card">● Gemini API connected<br>'
        '<span style="opacity:0.7;font-size:0.85rem;">'
        'All systems operational'
        '</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="power-card">'
        '<div style="font-size:2rem;">💎</div>'
        '<div class="power-title">Gemini Power</div>'
        '<br>'
        '<div style="opacity:0.7;font-size:0.85rem;">'
        'Fast. Smart. Local UI.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 7. HEADER
# =========================================================

st.markdown(
    '<p class="gemini-title">✦ Gemini Local</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="gemini-subtitle">'
    'Your local AI assistant powered by Gemini'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="gradient-line"></div>',
    unsafe_allow_html=True
)


# =========================================================
# 8. SUGGESTION BUTTONS
# =========================================================

if len(st.session_state.messages) == 0:

    st.markdown("## ✨ How can I help you today?")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🧠 Explain Transformers",
            use_container_width=True
        ):
            st.session_state.pending_prompt = (
                "Explain Transformer architecture "
                "in a simple and understandable way."
            )
            st.rerun()

        if st.button(
            "📚 Explain RAG",
            use_container_width=True
        ):
            st.session_state.pending_prompt = (
                "Explain Retrieval Augmented Generation "
                "with a simple example."
            )
            st.rerun()

    with col2:

        if st.button(
            "🎓 Explain LLM Training",
            use_container_width=True
        ):
            st.session_state.pending_prompt = (
                "Explain how modern LLMs are trained."
            )
            st.rerun()

        if st.button(
            "💻 Generate Python Code",
            use_container_width=True
        ):
            st.session_state.pending_prompt = (
                "Give me a useful Python project "
                "for learning Generative AI."
            )
            st.rerun()


# =========================================================
# 9. DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    avatar = (
        "🧑"
        if message["role"] == "user"
        else "✨"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.markdown(message["content"])


# =========================================================
# 10. USER INPUT
# =========================================================

user_prompt = st.chat_input(
    "Ask Gemini anything..."
)

if st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# =========================================================
# 11. HANDLE CHAT
# =========================================================

if user_prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message(
        "user",
        avatar="🧑"
    ):
        st.markdown(user_prompt)


    # Build full conversation history
    conversation = []

    for msg in st.session_state.messages:

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
                        "text": msg["content"]
                    }
                ]
            }
        )


    # Generate Gemini response
    with st.chat_message(
        "assistant",
        avatar="✨"
    ):

        placeholder = st.empty()

        full_response = ""

        try:

            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=conversation
            )

            for chunk in stream:

                if chunk.text:
                    full_response += chunk.text

                    placeholder.markdown(
                        full_response + "▌"
                    )

            placeholder.markdown(
                full_response
            )

        except Exception as e:

            full_response = (
                "There was an error communicating "
                "with Gemini.\n\n"
                f"{e}"
            )

            placeholder.error(
                full_response
            )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )