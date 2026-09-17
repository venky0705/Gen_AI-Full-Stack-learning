import os

import streamlit as st
from dotenv import load_dotenv

from rag import PersonalRAG


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Venkatesh Vemala | AI Profile",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(150,150,150,0.15);
}

[data-testid="stChatMessage"] {
    border: 1px solid rgba(150,150,150,0.12);
    border-radius: 18px;
    padding: 0.5rem;
    margin-bottom: 0.8rem;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 44px;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(150,150,150,0.15);
    padding: 14px;
    border-radius: 15px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# ENV
# ============================================================

load_dotenv()


def get_secret(
    name,
    default=None
):

    try:

        if name in st.secrets:
            return st.secrets[name]

    except Exception:
        pass


    return os.getenv(
        name,
        default
    )


# ============================================================
# GROQ KEY
# ============================================================

GROQ_API_KEY = get_secret(
    "GROQ_API_KEY"
)


GROQ_MODEL = get_secret(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY is missing."
    )

    st.info(
        "Add GROQ_API_KEY to your .env file locally "
        "or Streamlit Secrets in the cloud."
    )

    st.stop()


# Make key available to rag.py
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

os.environ["GROQ_MODEL"] = GROQ_MODEL


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "pending_question" not in st.session_state:

    st.session_state.pending_question = None


# ============================================================
# LOAD RAG ENGINE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_rag():

    return PersonalRAG()


try:

    with st.spinner(
        "Loading Venkatesh's profile..."
    ):

        rag = load_rag()


except Exception as error:

    st.error(
        "The RAG system could not start."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🧠 Venkatesh AI"
    )


    st.caption(
        "Interactive Professional Profile"
    )


    st.divider()


    st.markdown(
        """
### Explore

💼 Professional Experience

🎓 Education

🤖 AI & Machine Learning

📚 Academic Modules

🚀 Projects

🛠 Technical Skills

🏆 Certifications

🌍 Languages
"""
    )


    st.divider()


    st.markdown(
        f"""
**Embedding**

BAAI/bge-small-en-v1.5


**Vector DB**

FAISS


**Generator**

Groq · {GROQ_MODEL}
"""
    )


    st.divider()


    if st.button(
        "🗑️ New conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "👋 Meet Venkatesh Vemala"
)


st.subheader(
    "AI Engineer • Data Science • Generative AI"
)


st.write(
    """
Ask about Venkatesh's professional experience,
education, academic background, projects,
technical skills and certifications.
"""
)


st.caption(
    "Powered by Retrieval-Augmented Generation."
)


st.divider()


# ============================================================
# PROFILE CARDS
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Profile",
        "AI Engineer"
    )


with col2:

    st.metric(
        "Focus",
        "Gen AI / ML"
    )


with col3:

    st.metric(
        "Education",
        "M.Sc. Data Science"
    )


with col4:

    st.metric(
        "Location",
        "Germany"
    )


st.write("")


# ============================================================
# STARTER QUESTIONS
# ============================================================

if not st.session_state.messages:

    st.subheader(
        "Ask about Venkatesh"
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        if st.button(
            "💼 Work experience",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What is Venkatesh's work experience?"
            )


    with c2:

        if st.button(
            "🎓 Education",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "Tell me about Venkatesh's education."
            )


    with c3:

        if st.button(
            "🤖 AI skills",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What are Venkatesh's AI and machine learning skills?"
            )


    with c4:

        if st.button(
            "🚀 Projects",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What projects has Venkatesh worked on?"
            )


    c5, c6, c7, c8 = (
        st.columns(4)
    )


    with c5:

        if st.button(
            "📚 Master's subjects",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What subjects has Venkatesh studied in his master's?"
            )


    with c6:

        if st.button(
            "🏥 Siemens",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What does Venkatesh do at Siemens Healthineers?"
            )


    with c7:

        if st.button(
            "☁️ Accenture",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What did Venkatesh do at Accenture?"
            )


    with c8:

        if st.button(
            "🏆 Certifications",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What certifications does Venkatesh have?"
            )


    st.divider()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# INPUT
# ============================================================

typed_question = (
    st.chat_input(
        "Ask something about Venkatesh..."
    )
)


if typed_question:

    user_question = (
        typed_question.strip()
    )


elif st.session_state.pending_question:

    user_question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None


else:

    user_question = None


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_question:


    st.session_state.messages.append(
        {
            "role": "user",

            "content": user_question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching Venkatesh's profile..."
        ):

            try:

                result = rag.ask(

                    question=user_question,

                    messages=(
                        st.session_state
                        .messages[:-1]
                    )
                )


                answer = result[
                    "answer"
                ]


                source_documents = result[
                    "sources"
                ]


                context_length = result[
                    "context_length"
                ]


                st.markdown(
                    answer
                )


                # ============================================
                # SAFE SOURCE VIEW
                # ============================================

                with st.expander(
                    "📖 Knowledge sources"
                ):

                    pages = []


                    for doc in (
                        source_documents
                    ):

                        page = (
                            doc.metadata.get(
                                "page_number"
                            )
                        )


                        if page is None:

                            raw_page = (
                                doc.metadata.get(
                                    "page"
                                )
                            )


                            if isinstance(
                                raw_page,
                                int
                            ):

                                page = (
                                    raw_page + 1
                                )


                        if (
                            page
                            and page not in pages
                        ):

                            pages.append(
                                page
                            )


                    if pages:

                        pages = sorted(
                            pages
                        )


                        st.write(
                            "Knowledge-base pages used:"
                        )


                        st.write(
                            ", ".join(
                                str(page)
                                for page in pages
                            )
                        )


                    else:

                        st.write(
                            "Information retrieved from "
                            "Venkatesh's knowledge base."
                        )


            except Exception as error:

                error_text = str(
                    error
                )


                # --------------------------------------------
                # GROQ RATE LIMIT
                # --------------------------------------------

                if (
                    "429" in error_text
                    or "rate_limit" in error_text.lower()
                    or "rate limit" in error_text.lower()
                ):

                    answer = (
                        "The AI service has reached its "
                        "temporary usage limit. "
                        "Please try again shortly."
                    )

                    st.warning(
                        answer
                    )


                else:

                    answer = (
                        "The profile assistant encountered "
                        "an error while processing this question."
                    )

                    st.error(
                        answer
                    )


                    with st.expander(
                        "Developer error details"
                    ):

                        st.exception(
                            error
                        )


    st.session_state.messages.append(
        {
            "role": "assistant",

            "content": answer
        }
    )