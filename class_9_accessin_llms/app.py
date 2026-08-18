import os
import io
import base64
import tempfile
import wave

import streamlit as st
from dotenv import load_dotenv

from google import genai
from google.genai import types

from groq import Groq

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Free Multimodal AI Playground",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD API KEYS
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


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


# ============================================================
# MODELS
# ============================================================

GEMINI_TEXT_MODEL = "gemini-3.5-flash-lite"

GEMINI_VISION_MODEL = "gemini-3.6-flash"

GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"

GROQ_TEXT_MODEL = "openai/gpt-oss-20b"

GROQ_STT_MODEL = "whisper-large-v3-turbo"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 10%,
                rgba(91, 33, 182, 0.15),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #071426 0%,
                #080d1b 55%,
                #110c20 100%
            );
    }


    /* Main content */

    .block-container {
        max-width: 1450px;
        padding-top: 3rem;
        padding-bottom: 7rem;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07162c 0%,
                #081225 100%
            );

        border-right: 1px solid rgba(255,255,255,0.10);
    }


    /* Headings */

    h1, h2, h3 {
        color: #ffffff;
    }


    /* Caption */

    .stCaption {
        color: #8f9aad;
    }


    /* Chat message */

    [data-testid="stChatMessage"] {

        background:
            linear-gradient(
                135deg,
                rgba(21, 32, 60, 0.72),
                rgba(20, 17, 43, 0.75)
            );

        border: 1px solid rgba(103, 126, 234, 0.20);

        border-radius: 18px;

        padding: 10px;

        margin-bottom: 15px;

        box-shadow:
            0px 8px 25px rgba(0,0,0,0.15);

        backdrop-filter: blur(12px);
    }


    /* Inputs */

    textarea,
    input {

        border-radius: 14px !important;
    }


    /* Buttons */

    .stButton > button {

        border-radius: 12px;

        border:
            1px solid rgba(
                103,
                126,
                234,
                0.35
            );

        transition: 0.2s;
    }


    .stButton > button:hover {

        border-color: #7c83ff;

        box-shadow:
            0 0 15px
            rgba(99,102,241,0.25);

        transform: translateY(-1px);
    }


    /* File uploader */

    [data-testid="stFileUploader"] {

        border-radius: 16px;

        padding: 8px;
    }


    /* Divider */

    hr {
        border-color:
            rgba(255,255,255,0.12);
    }


    /* PDF status card */

    .pdf-card {

        padding: 18px;

        border-radius: 16px;

        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 70, 0.80),
                rgba(25, 20, 48, 0.80)
            );

        border:
            1px solid
            rgba(99,102,241,0.25);

        margin-bottom: 20px;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "text_messages" not in st.session_state:
    st.session_state.text_messages = []


if "pdf_messages" not in st.session_state:
    st.session_state.pdf_messages = []


if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []


if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None


if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ============================================================
# LOCAL EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# PDF FUNCTIONS
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text
                }
            )

    return pages


def split_text(
    pages,
    chunk_size=1200,
    overlap=200
):

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append(
                {
                    "page": page["page"],
                    "text": chunk_text
                }
            )

            start += chunk_size - overlap

    return chunks


def build_pdf_index(chunks):

    embedding_model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def retrieve_pdf_chunks(
    question,
    top_k=5
):

    if st.session_state.pdf_index is None:
        return []


    embedding_model = load_embedding_model()

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = (
        st.session_state.pdf_index.search(
            query_embedding,
            top_k
        )
    )


    results = []

    for index in indices[0]:

        if index == -1:
            continue

        results.append(
            st.session_state.pdf_chunks[index]
        )

    return results


# ============================================================
# TEXT LLM
# ============================================================

def generate_groq_response(messages):

    if groq_client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing."
        )


    response = groq_client.chat.completions.create(

        model=GROQ_TEXT_MODEL,

        messages=messages,

        temperature=0.4,

        max_tokens=2000
    )


    return response.choices[0].message.content


def generate_gemini_response(messages):

    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing."
        )


    conversation = ""

    for message in messages:

        conversation += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )


    response = gemini_client.models.generate_content(

        model=GEMINI_TEXT_MODEL,

        contents=conversation
    )


    return response.text


# ============================================================
# PDF QUESTION ANSWERING
# ============================================================

def answer_pdf_question(
    question,
    provider
):

    relevant_chunks = retrieve_pdf_chunks(
        question,
        top_k=5
    )


    if not relevant_chunks:

        return (
            "I could not find relevant "
            "content in the uploaded PDF."
        )


    context_parts = []

    for chunk in relevant_chunks:

        context_parts.append(

            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"

        )


    context = "\n\n".join(
        context_parts
    )


    prompt = f"""
You are answering questions about an uploaded PDF.

Use ONLY the provided PDF context.

If the answer cannot be found in the context, say:

"I could not find that information in the uploaded PDF."

Do not invent information.

When possible, mention the page number where the information came from.

PDF CONTEXT:

{context}


USER QUESTION:

{question}


ANSWER:
"""


    if provider == "Groq":

        messages = [

            {
                "role": "system",
                "content":
                    "Answer questions using only "
                    "the supplied PDF context."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]


        return generate_groq_response(
            messages
        )


    else:

        response = gemini_client.models.generate_content(

            model=GEMINI_TEXT_MODEL,

            contents=prompt
        )

        return response.text


# ============================================================
# IMAGE → TEXT
# ============================================================

def describe_image(
    uploaded_image,
    prompt
):

    image_bytes = uploaded_image.getvalue()


    response = gemini_client.models.generate_content(

        model=GEMINI_VISION_MODEL,

        contents=[

            prompt,

            types.Part.from_bytes(
                data=image_bytes,
                mime_type=uploaded_image.type
            )

        ]
    )


    return response.text


# ============================================================
# AUDIO → TEXT
# ============================================================

def transcribe_audio(
    uploaded_audio
):

    suffix = ".wav"

    if uploaded_audio.name:

        extension = os.path.splitext(
            uploaded_audio.name
        )[1]

        if extension:
            suffix = extension


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            uploaded_audio.getvalue()
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

                    model=GROQ_STT_MODEL,

                    response_format="text"
                )
            )


        return transcription


    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "API Status"
    )


    if GEMINI_API_KEY:

        st.success(
            "Google Gemini API connected"
        )

    else:

        st.error(
            "Gemini API not connected"
        )


    if GROQ_API_KEY:

        st.success(
            "Groq API connected"
        )

    else:

        st.error(
            "Groq API not connected"
        )


    st.divider()


    st.subheader(
        "Available models"
    )


    st.write(
        "**Gemini text:**"
    )

    st.code(
        GEMINI_TEXT_MODEL
    )


    st.write(
        "**Gemini vision:**"
    )

    st.code(
        GEMINI_VISION_MODEL
    )


    st.write(
        "**Groq text:**"
    )

    st.code(
        GROQ_TEXT_MODEL
    )


    st.write(
        "**Groq speech-to-text:**"
    )

    st.code(
        GROQ_STT_MODEL
    )


    st.write(
        "**Gemini TTS:**"
    )

    st.code(
        GEMINI_TTS_MODEL
    )


    st.divider()


    st.write(
        "**PDF embeddings:**"
    )

    st.code(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "## ✨ Free Multimodal AI Playground"
)

st.caption(
    "Gemini + Groq | Text, Image, Audio and PDF"
)


# ============================================================
# TASK SELECTOR
# ============================================================

task = st.selectbox(

    "Choose task",

    [
        "Text → Text",
        "Image → Text",
        "Audio → Text",
        "PDF → Chat"
    ]
)


# ============================================================
# TEXT → TEXT
# ============================================================

if task == "Text → Text":

    provider = st.selectbox(

        "Provider",

        [
            "Groq",
            "Gemini"
        ]
    )


    col1, col2 = st.columns(
        [8, 1]
    )


    with col1:

        st.caption(
            f"Conversation using {provider}"
        )


    with col2:

        if st.button(
            "Clear",
            key="clear_text"
        ):

            st.session_state.text_messages = []

            st.rerun()


    for message in st.session_state.text_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    prompt = st.chat_input(
        f"Message {provider}...",
        key="text_chat"
    )


    if prompt:

        st.session_state.text_messages.append(

            {
                "role": "user",
                "content": prompt
            }

        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt
            )


        try:

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Thinking..."
                ):

                    if provider == "Groq":

                        response = generate_groq_response(
                            st.session_state.text_messages
                        )

                    else:

                        response = generate_gemini_response(
                            st.session_state.text_messages
                        )


                st.markdown(
                    response
                )


            st.session_state.text_messages.append(

                {
                    "role": "assistant",
                    "content": response
                }

            )


        except Exception as e:

            st.error(
                str(e)
            )


# ============================================================
# IMAGE → TEXT
# ============================================================

elif task == "Image → Text":

    st.subheader(
        "🖼️ Image Understanding"
    )


    image_file = st.file_uploader(

        "Upload an image",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    image_prompt = st.text_area(

        "What should Gemini do?",

        value="Describe this image in detail."
    )


    if image_file:

        st.image(
            image_file,
            width=500
        )


        if st.button(
            "Analyze image",
            use_container_width=True
        ):

            if gemini_client is None:

                st.error(
                    "Gemini API is not configured."
                )

            else:

                try:

                    with st.spinner(
                        "Analyzing image..."
                    ):

                        response = describe_image(
                            image_file,
                            image_prompt
                        )


                    st.markdown(
                        "### Response"
                    )

                    st.markdown(
                        response
                    )


                except Exception as e:

                    st.error(
                        str(e)
                    )


# ============================================================
# AUDIO → TEXT
# ============================================================

elif task == "Audio → Text":

    st.subheader(
        "🎤 Speech to Text"
    )


    audio_file = st.file_uploader(

        "Upload audio",

        type=[
            "wav",
            "mp3",
            "m4a",
            "ogg"
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

            if groq_client is None:

                st.error(
                    "Groq API is not configured."
                )

            else:

                try:

                    with st.spinner(
                        "Transcribing..."
                    ):

                        transcription = transcribe_audio(
                            audio_file
                        )


                    st.markdown(
                        "### Transcription"
                    )

                    st.write(
                        transcription
                    )


                except Exception as e:

                    st.error(
                        str(e)
                    )


# ============================================================
# PDF → CHAT
# ============================================================

elif task == "PDF → Chat":

    st.subheader(
        "📄 Chat with your PDF"
    )


    st.caption(
        "Upload a PDF, then ask questions about its content."
    )


    provider = st.selectbox(

        "Answer using",

        [
            "Groq",
            "Gemini"
        ],

        key="pdf_provider"
    )


    uploaded_pdf = st.file_uploader(

        "Upload PDF",

        type=["pdf"],

        key="pdf_upload"
    )


    # --------------------------------------------------------
    # PROCESS PDF
    # --------------------------------------------------------

    if uploaded_pdf is not None:

        current_pdf = (
            uploaded_pdf.name
        )


        if (
            st.session_state.pdf_name
            != current_pdf
        ):

            try:

                with st.spinner(
                    "Reading and indexing PDF..."
                ):

                    pages = extract_pdf_text(
                        uploaded_pdf
                    )


                    if not pages:

                        st.error(
                            "No extractable text "
                            "was found in this PDF."
                        )

                        st.stop()


                    chunks = split_text(
                        pages
                    )


                    index = build_pdf_index(
                        chunks
                    )


                    st.session_state.pdf_chunks = (
                        chunks
                    )

                    st.session_state.pdf_index = (
                        index
                    )

                    st.session_state.pdf_name = (
                        current_pdf
                    )

                    st.session_state.pdf_messages = []


            except Exception as e:

                st.error(
                    f"Could not process PDF: {e}"
                )

                st.stop()


        st.markdown(

            f"""
            <div class="pdf-card">

            <b>📄 PDF loaded</b><br><br>

            File: {st.session_state.pdf_name}<br>

            Chunks indexed:
            {len(st.session_state.pdf_chunks)}

            </div>
            """,

            unsafe_allow_html=True
        )


        col1, col2 = st.columns(
            [8, 1]
        )


        with col1:

            st.caption(
                f"Ask questions using {provider}"
            )


        with col2:

            if st.button(
                "Clear",
                key="clear_pdf"
            ):

                st.session_state.pdf_messages = []

                st.rerun()


        # ----------------------------------------------------
        # SHOW PDF CONVERSATION
        # ----------------------------------------------------

        for message in st.session_state.pdf_messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )


        # ----------------------------------------------------
        # PDF CHAT INPUT
        # ----------------------------------------------------

        pdf_question = st.chat_input(

            "Ask something about the PDF...",

            key="pdf_chat"
        )


        if pdf_question:

            st.session_state.pdf_messages.append(

                {
                    "role": "user",
                    "content": pdf_question
                }

            )


            with st.chat_message(
                "user"
            ):

                st.markdown(
                    pdf_question
                )


            try:

                with st.chat_message(
                    "assistant"
                ):

                    with st.spinner(
                        "Searching PDF..."
                    ):

                        answer = answer_pdf_question(
                            pdf_question,
                            provider
                        )


                    st.markdown(
                        answer
                    )


                st.session_state.pdf_messages.append(

                    {
                        "role": "assistant",
                        "content": answer
                    }

                )


            except Exception as e:

                st.error(
                    str(e)
                )


    else:

        st.info(
            "Upload a PDF to start chatting with it."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Free API access is subject to provider quotas and rate limits."
)