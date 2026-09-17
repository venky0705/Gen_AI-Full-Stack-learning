# ============================================================
# VENKATESH VEMALA - PUBLIC PERSONAL RAG CHATBOT
#
# Architecture:
#
# Personal PDF
#      ↓
# PyPDFLoader
#      ↓
# Text Chunking
#      ↓
# Hugging Face BGE Embeddings
#      ↓
# FAISS
#      ↓
# MMR Retrieval
#      ↓
# Gemini
#      ↓
# Public Streamlit Chatbot
# ============================================================


import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Venkatesh Vemala | AI Profile",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. UI CSS
# ============================================================

st.markdown(
    """
<style>

/* ----------------------------------------------------------
   APP BACKGROUND
---------------------------------------------------------- */

[data-testid="stAppViewContainer"] {

    background:
        radial-gradient(
            circle at 15% 0%,
            rgba(85, 80, 255, 0.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 5%,
            rgba(170, 70, 255, 0.08),
            transparent 30%
        );
}


/* ----------------------------------------------------------
   MAIN CONTENT
---------------------------------------------------------- */

.block-container {

    max-width: 1200px;

    padding-top: 2rem;

    padding-bottom: 7rem;
}


/* ----------------------------------------------------------
   SIDEBAR
---------------------------------------------------------- */

[data-testid="stSidebar"] {

    border-right:
        1px solid rgba(150,150,150,0.15);
}


/* ----------------------------------------------------------
   CHAT MESSAGES
---------------------------------------------------------- */

[data-testid="stChatMessage"] {

    border:
        1px solid rgba(150,150,150,0.12);

    border-radius: 18px;

    padding: 0.5rem;

    margin-bottom: 0.8rem;
}


/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

.stButton > button {

    border-radius: 12px;

    min-height: 45px;

    font-weight: 500;

    border:
        1px solid rgba(150,150,150,0.20);
}


/* ----------------------------------------------------------
   METRICS
---------------------------------------------------------- */

div[data-testid="stMetric"] {

    border:
        1px solid rgba(150,150,150,0.15);

    padding: 16px;

    border-radius: 16px;

    background:
        rgba(255,255,255,0.025);
}


/* ----------------------------------------------------------
   EXPANDERS
---------------------------------------------------------- */

div[data-testid="stExpander"] {

    border-radius: 14px;
}


/* ----------------------------------------------------------
   CHAT INPUT
---------------------------------------------------------- */

[data-testid="stChatInput"] {

    border-radius: 18px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


def get_secret(name, default=None):

    """
    Works both locally and on Streamlit Cloud.
    """

    # Streamlit Cloud
    try:

        value = st.secrets.get(name)

        if value:
            return value

    except Exception:
        pass


    # Local .env
    return os.getenv(
        name,
        default,
    )


GOOGLE_API_KEY = get_secret(
    "GOOGLE_API_KEY"
)


if not GOOGLE_API_KEY:

    st.error(
        "GOOGLE_API_KEY is missing."
    )

    st.info(
        "Add GOOGLE_API_KEY to your .env file locally "
        "or Streamlit Secrets when deploying."
    )

    st.stop()


# ============================================================
# 4. FILE PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


PDF_PATH = (
    BASE_DIR
    / "Venkatesh_Vemala_Personal_RAG_Knowledge_Base.pdf"
)


FAISS_PATH = (
    BASE_DIR
    / "faiss_personal_index"
)


if not PDF_PATH.exists():

    st.error(
        "Knowledge-base PDF was not found."
    )

    st.code(
        str(PDF_PATH)
    )

    st.stop()


# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================

# Free local Hugging Face embedding model
EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)


# Gemini is only responsible for final answer generation
GEMINI_MODEL = get_secret(
    "GEMINI_CHAT_MODEL",
    "gemini-3.6-flash",
)


# ============================================================
# 6. INTERNAL RAG SETTINGS
#
# These are NOT shown to public users.
# ============================================================

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200

TOP_K = 14

FETCH_K = 40

MMR_LAMBDA = 0.70

MAX_CONTEXT_CHARS = 50000

HISTORY_MESSAGES = 6


# ============================================================
# 7. SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "pending_question" not in st.session_state:

    st.session_state.pending_question = None


# ============================================================
# 8. LOAD HUGGING FACE EMBEDDINGS
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


# ============================================================
# 9. CREATE / LOAD FAISS INDEX
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_vector_store():

    embeddings = (
        load_embeddings()
    )


    # --------------------------------------------------------
    # LOAD SAVED INDEX
    # --------------------------------------------------------

    if FAISS_PATH.exists():

        try:

            vector_store = (
                FAISS.load_local(

                    str(
                        FAISS_PATH
                    ),

                    embeddings,

                    allow_dangerous_deserialization=True,
                )
            )

            return vector_store


        except Exception:

            # If embedding/index mismatch occurs,
            # rebuild automatically.

            shutil.rmtree(
                FAISS_PATH,
                ignore_errors=True,
            )


    # --------------------------------------------------------
    # LOAD PDF
    # --------------------------------------------------------

    loader = PyPDFLoader(
        str(PDF_PATH)
    )


    documents = (
        loader.load()
    )


    # --------------------------------------------------------
    # PAGE METADATA
    # --------------------------------------------------------

    for document in documents:

        page = (
            document
            .metadata
            .get("page")
        )


        if isinstance(
            page,
            int,
        ):

            document.metadata[
                "page_number"
            ] = page + 1


    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=[

                "\n\n",

                "\n",

                ". ",

                "; ",

                " ",

                "",
            ],
        )
    )


    chunks = (
        splitter
        .split_documents(
            documents
        )
    )


    # --------------------------------------------------------
    # CHUNK METADATA
    # --------------------------------------------------------

    for index, chunk in enumerate(
        chunks
    ):

        chunk.metadata[
            "chunk_id"
        ] = index


    # --------------------------------------------------------
    # CREATE FAISS
    # --------------------------------------------------------

    vector_store = (
        FAISS.from_documents(

            documents=chunks,

            embedding=embeddings,
        )
    )


    # --------------------------------------------------------
    # SAVE INDEX
    # --------------------------------------------------------

    vector_store.save_local(
        str(FAISS_PATH)
    )


    return vector_store


# ============================================================
# 10. LOAD GEMINI
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_llm():

    return ChatGoogleGenerativeAI(

        model=GEMINI_MODEL,

        google_api_key=GOOGLE_API_KEY,

        temperature=0.1,
    )


# ============================================================
# 11. CLEAN GEMINI RESPONSE
# ============================================================

def extract_gemini_text(
    response
):

    """
    Gemini may return either:

    "normal text"

    OR

    [
        {
            "type": "text",
            "text": "...",
            "extras": {...}
        }
    ]

    We only return the text.
    """

    content = (
        response.content
    )


    # --------------------------------------------------------
    # STRING RESPONSE
    # --------------------------------------------------------

    if isinstance(
        content,
        str,
    ):

        return content.strip()


    # --------------------------------------------------------
    # BLOCK RESPONSE
    # --------------------------------------------------------

    if isinstance(
        content,
        list,
    ):

        text_parts = []


        for block in content:


            if isinstance(
                block,
                dict,
            ):

                if (
                    block.get("type")
                    == "text"
                ):

                    text = (
                        block.get(
                            "text",
                            "",
                        )
                    )


                    if text:

                        text_parts.append(
                            text
                        )


            elif isinstance(
                block,
                str,
            ):

                text_parts.append(
                    block
                )


        return "\n\n".join(
            text_parts
        ).strip()


    return str(
        content
    ).strip()


# ============================================================
# 12. CHAT HISTORY
# ============================================================

def format_history(
    messages
):

    if not messages:

        return (
            "No previous conversation."
        )


    recent_messages = (
        messages[
            -HISTORY_MESSAGES:
        ]
    )


    history = []


    for message in recent_messages:

        role = (
            message.get(
                "role"
            )
        )

        text = (
            message.get(
                "content",
                "",
            )
        )


        if role == "user":

            label = "Visitor"

        else:

            label = "Assistant"


        history.append(
            f"{label}: {text}"
        )


    return "\n".join(
        history
    )


# ============================================================
# 13. PREVIOUS QUESTION
# ============================================================

def get_previous_user_question():

    # Current question has already been
    # appended, so skip the newest message.

    messages = (
        st.session_state
        .messages[:-1]
    )


    for message in reversed(
        messages
    ):

        if (
            message.get("role")
            == "user"
        ):

            return message.get(
                "content",
                "",
            )


    return ""


# ============================================================
# 14. RETRIEVAL
# ============================================================

def retrieve_documents(
    question
):

    previous_question = (
        get_previous_user_question()
    )


    # --------------------------------------------------------
    # FOLLOW-UP QUERY SUPPORT
    # --------------------------------------------------------

    if previous_question:

        retrieval_query = f"""
The visitor is asking about Venkatesh Vemala.

Previous question:
{previous_question}

Current question:
{question}

Retrieve information about Venkatesh that helps answer
the current question.
"""


    else:

        retrieval_query = f"""
The visitor is asking about Venkatesh Vemala.

Question:
{question}
"""


    # --------------------------------------------------------
    # MMR RETRIEVAL
    # --------------------------------------------------------

    documents = (
        vector_store
        .max_marginal_relevance_search(

            query=retrieval_query,

            k=TOP_K,

            fetch_k=FETCH_K,

            lambda_mult=MMR_LAMBDA,
        )
    )


    return documents


# ============================================================
# 15. BUILD RETRIEVED CONTEXT
# ============================================================

def build_context(
    documents
):

    context_parts = []

    used_documents = []

    current_length = 0

    seen = set()


    for document in documents:

        text = (
            document
            .page_content
            .strip()
        )


        if not text:
            continue


        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        normalized = " ".join(
            text.lower().split()
        )


        if normalized in seen:
            continue


        seen.add(
            normalized
        )


        page = (
            document
            .metadata
            .get(
                "page_number"
            )
        )


        if page is None:

            raw_page = (
                document
                .metadata
                .get("page")
            )


            if isinstance(
                raw_page,
                int,
            ):

                page = (
                    raw_page + 1
                )

            else:

                page = "Unknown"


        chunk_id = (
            document
            .metadata
            .get(
                "chunk_id",
                "Unknown",
            )
        )


        formatted = f"""
[Knowledge Source - Page {page}]

{text}
"""


        # ----------------------------------------------------
        # CONTEXT LIMIT
        # ----------------------------------------------------

        if (
            current_length
            + len(formatted)
            > MAX_CONTEXT_CHARS
        ):

            remaining = (
                MAX_CONTEXT_CHARS
                - current_length
            )


            if remaining > 500:

                context_parts.append(
                    formatted[
                        :remaining
                    ]
                )

                used_documents.append(
                    document
                )


            break


        context_parts.append(
            formatted
        )


        used_documents.append(
            document
        )


        current_length += (
            len(formatted)
        )


    context = "\n\n".join(
        context_parts
    )


    return (
        context,
        used_documents,
        len(context),
    )


# ============================================================
# 16. PUBLIC CHATBOT PROMPT
# ============================================================

PUBLIC_RAG_PROMPT = """
You are the public professional profile assistant for
Venkatesh Vemala.

People visiting this application are learning about Venkatesh.

Therefore, ALWAYS speak about Venkatesh in THIRD PERSON.

For example:

Correct:
"Venkatesh is pursuing an M.Sc. in Data Science at FAU."

Incorrect:
"You are pursuing an M.Sc. in Data Science."

Correct:
"Venkatesh's current average grade is 2.9."

Incorrect:
"Your current average grade is 2.9."


============================================================
IDENTITY HANDLING
============================================================

Questions may contain words such as:

"my"
"me"
"I"
"his"
"him"
"Venkatesh"
"Vemala Venkatesh"

In this application these questions are intended to ask
about Venkatesh Vemala.

However, the final answer must ALWAYS refer to him as:

"Venkatesh"

or

"Venkatesh Vemala"

or use appropriate third-person pronouns such as:

"he"
"his"


============================================================
FACTUAL GROUNDING
============================================================

Use ONLY information supported by the retrieved knowledge base.

Do not invent facts.

Do not guess.

Do not fill missing information using general knowledge.

If requested information is absent, say:

"That information is not specified in Venkatesh's available knowledge base."

Do not say:

"I could not find your information."

Do not address the visitor as if the visitor is Venkatesh.


============================================================
IMPORTANT DISTINCTION
============================================================

Do not confuse bachelor's and master's information.

If the knowledge base gives:

Bachelor's degree:
B.E. Computer Science

but does not provide the bachelor's final grade,
say that the bachelor's final grade is not specified.

Do NOT use Venkatesh's master's average as his bachelor's grade.


============================================================
WORK EXPERIENCE
============================================================

When asked about Venkatesh's work experience, present supported
information such as:

- Company
- Role
- Dates
- Location
- Responsibilities
- Technologies
- Achievements


============================================================
EDUCATION
============================================================

When asked about education, include supported information such as:

- Institution
- Degree
- Specialization
- Dates
- Location


============================================================
ACADEMIC RESULTS
============================================================

For examination questions, include supported information such as:

- Module
- Grade
- ECTS
- Status

When talking about the master's average, clearly state whether
it is current/preliminary rather than a final degree grade,
if the context says so.


============================================================
PROJECTS
============================================================

When asked about projects, include supported information such as:

- Project title
- Problem
- Technologies
- Implementation
- Result


============================================================
SKILLS
============================================================

When asked about skills, group them naturally.

Possible categories may include:

- Generative AI / NLP
- Machine Learning
- Deep Learning
- Data Engineering
- Cloud
- Analytics
- Programming

Only include skills present in the retrieved context.


============================================================
ANSWER STYLE
============================================================

Use concise answers for simple questions.

Example:

Question:
"What is Venkatesh's master's grade?"

Good answer:

"Venkatesh's current average grade in the M.Sc. Data Science
program at FAU is 2.9. This is a preliminary average based on
the completed credits and is not yet his final degree grade."


Use structured answers for broad questions.

Example:

Question:
"What is Venkatesh's work experience?"

Use company headings and bullet points.


============================================================
PUBLIC PRIVACY RULES
============================================================

This chatbot is public.

Even if retrieved text contains private identifiers, NEVER expose:

- street or residential address
- apartment number
- student registration number
- private identification numbers
- hidden credentials
- API keys
- authentication information

If asked for such information, respond:

"That information is not provided through this public profile."


============================================================
CONVERSATION
============================================================

Recent conversation may help interpret follow-up questions.

However, factual answers must still be supported by retrieved
knowledge-base information.


============================================================
RECENT CONVERSATION
============================================================

{history}


============================================================
RETRIEVED KNOWLEDGE BASE
============================================================

{context}


============================================================
VISITOR QUESTION
============================================================

{question}


============================================================
PUBLIC PROFILE ANSWER
============================================================
"""


prompt_template = PromptTemplate(

    template=PUBLIC_RAG_PROMPT,

    input_variables=[
        "history",
        "context",
        "question",
    ],
)


# ============================================================
# 17. ASK RAG
# ============================================================

def ask_rag(
    question
):

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    retrieved_documents = (
        retrieve_documents(
            question
        )
    )


    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    (
        context,
        used_documents,
        context_length,
    ) = build_context(
        retrieved_documents
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = format_history(

        st.session_state
        .messages[:-1]
    )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    final_prompt = (
        prompt_template
        .format(

            history=history,

            context=context,

            question=question,
        )
    )


    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    response = (
        llm.invoke(
            final_prompt
        )
    )


    # --------------------------------------------------------
    # CLEAN OUTPUT
    # --------------------------------------------------------

    answer = (
        extract_gemini_text(
            response
        )
    )


    return (
        answer,
        used_documents,
        context_length,
    )


# ============================================================
# 18. LOAD RAG SYSTEM
# ============================================================

with st.spinner(
    "Loading Venkatesh's profile..."
):

    vector_store = (
        load_vector_store()
    )

    llm = (
        load_llm()
    )


# ============================================================
# 19. SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🧠 Venkatesh AI"
    )


    st.caption(
        "Interactive professional profile"
    )


    st.divider()


    st.subheader(
        "About"
    )


    st.markdown(
        """
Explore Venkatesh Vemala's professional and academic background
through an AI-powered profile assistant.
"""
    )


    st.divider()


    st.subheader(
        "Explore"
    )


    st.markdown(
        """
💼 **Professional Experience**

🎓 **Education**

🤖 **AI & Machine Learning**

📚 **Academic Modules**

🚀 **Projects**

🛠️ **Technical Skills**

🏆 **Certifications**

🌍 **Languages**
"""
    )


    st.divider()


    st.caption(
        "Powered by Retrieval-Augmented Generation"
    )


    if st.button(
        "🗑️ New conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


# ============================================================
# 20. PUBLIC HEADER
# ============================================================

st.title(
    "👋 Meet Venkatesh Vemala"
)


st.subheader(
    "AI Engineer • Data Science • Generative AI"
)


st.markdown(
    """
Explore Venkatesh's professional experience, education,
AI projects, technical skills and academic background by
asking questions below.
"""
)


st.caption(
    "This assistant answers using a curated personal knowledge base."
)


st.divider()


# ============================================================
# 21. PROFILE SUMMARY CARDS
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        label="Profile",
        value="AI Engineer",
    )


with col2:

    st.metric(
        label="Focus",
        value="Gen AI / ML",
    )


with col3:

    st.metric(
        label="Education",
        value="M.Sc. Data Science",
    )


with col4:

    st.metric(
        label="Location",
        value="Germany",
    )


st.write("")


# ============================================================
# 22. STARTER QUESTIONS
# ============================================================

if not st.session_state.messages:

    st.subheader(
        "Ask about Venkatesh"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        if st.button(
            "💼 Work experience",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What is Venkatesh's work experience?"
            )


    with col2:

        if st.button(
            "🎓 Education",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "Tell me about Venkatesh's education."
            )


    with col3:

        if st.button(
            "🤖 AI skills",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What are Venkatesh's AI and machine learning skills?"
            )


    with col4:

        if st.button(
            "🚀 Projects",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What projects has Venkatesh worked on?"
            )


    col5, col6, col7, col8 = (
        st.columns(4)
    )


    with col5:

        if st.button(
            "📚 Master's subjects",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What subjects has Venkatesh studied in his master's?"
            )


    with col6:

        if st.button(
            "🏥 Siemens",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What does Venkatesh do at Siemens Healthineers?"
            )


    with col7:

        if st.button(
            "☁️ Accenture",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What did Venkatesh do at Accenture?"
            )


    with col8:

        if st.button(
            "🏆 Certifications",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What certifications does Venkatesh have?"
            )


    st.divider()


# ============================================================
# 23. CHAT HISTORY
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
# 24. CHAT INPUT
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


elif (
    st.session_state
    .pending_question
):

    user_question = (
        st.session_state
        .pending_question
    )

    st.session_state.pending_question = None


else:

    user_question = None


# ============================================================
# 25. PROCESS QUESTION
# ============================================================

if user_question:


    # --------------------------------------------------------
    # STORE VISITOR QUESTION
    # --------------------------------------------------------

    st.session_state.messages.append(

        {
            "role": "user",

            "content": user_question,
        }
    )


    # --------------------------------------------------------
    # DISPLAY QUESTION
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )


    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching Venkatesh's profile..."
        ):

            try:

                (
                    answer,
                    source_documents,
                    context_length,
                ) = ask_rag(
                    user_question
                )


                # --------------------------------------------
                # ANSWER
                # --------------------------------------------

                st.markdown(
                    answer
                )


                # --------------------------------------------
                # PUBLIC-SAFE SOURCE INFO
                #
                # We do NOT display raw retrieved chunks.
                # This prevents accidental private-data leaks.
                # --------------------------------------------

                with st.expander(
                    "📖 Knowledge sources"
                ):

                    pages = []


                    for document in (
                        source_documents
                    ):

                        page = (
                            document
                            .metadata
                            .get(
                                "page_number"
                            )
                        )


                        if page is None:

                            raw_page = (
                                document
                                .metadata
                                .get("page")
                            )


                            if isinstance(
                                raw_page,
                                int,
                            ):

                                page = (
                                    raw_page + 1
                                )


                        if (
                            page
                            and page
                            not in pages
                        ):

                            pages.append(
                                page
                            )


                    if pages:

                        pages = sorted(
                            pages
                        )


                        st.write(
                            "Information was retrieved "
                            "from knowledge-base page(s):"
                        )


                        st.write(
                            ", ".join(
                                str(page)
                                for page in pages
                            )
                        )


                    else:

                        st.write(
                            "Information was retrieved "
                            "from Venkatesh's personal "
                            "knowledge base."
                        )


            except Exception as error:

                answer = (
                    "The profile assistant encountered "
                    "an error while processing this question."
                )


                st.error(
                    answer
                )


                # --------------------------------------------
                # Only show detailed error locally
                # --------------------------------------------

                if (
                    os.getenv(
                        "DEBUG",
                        "false"
                    ).lower()
                    == "true"
                ):

                    with st.expander(
                        "Developer error"
                    ):

                        st.exception(
                            error
                        )


    # --------------------------------------------------------
    # SAVE RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(

        {
            "role": "assistant",

            "content": answer,
        }
    )