import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# ============================================================
# ENV
# ============================================================

load_dotenv()


def get_env_value(name, default=None):
    return os.getenv(name, default)


GROQ_API_KEY = get_env_value("GROQ_API_KEY")

GROQ_MODEL = get_env_value(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from environment variables."
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = (
    BASE_DIR
    / "Venkatesh_Vemala_Personal_RAG_Knowledge_Base.pdf"
)


if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}"
    )


# ============================================================
# RAG SETTINGS
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

TOP_K = 12
FETCH_K = 30

MAX_CONTEXT_CHARS = 40000

HISTORY_MESSAGES = 6


# ============================================================
# EMBEDDINGS
# ============================================================

def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ============================================================
# VECTOR STORE
# ============================================================

def build_vector_store():

    embeddings = load_embeddings()

    loader = PyPDFLoader(
        str(PDF_PATH)
    )

    documents = loader.load()


    for document in documents:

        page = document.metadata.get("page")

        if isinstance(page, int):
            document.metadata["page_number"] = page + 1


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            " ",
            ""
        ]
    )


    chunks = splitter.split_documents(
        documents
    )


    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i


    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


# ============================================================
# LLM
# ============================================================

def load_llm():

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1
    )


# ============================================================
# PROMPT
# ============================================================

RAG_TEMPLATE = """
You are the public professional profile assistant for
Venkatesh Vemala.

Visitors use this application to learn about Venkatesh.

Always speak about Venkatesh in THIRD PERSON.

Correct:
"Venkatesh is pursuing an M.Sc. in Data Science at FAU."

Incorrect:
"You are pursuing an M.Sc. in Data Science."


============================================================
FACTUAL RULES
============================================================

Use only the retrieved context.

Do not invent facts.

Do not guess missing information.

Do not use outside knowledge about Venkatesh.

If information is not present, say:

"That information is not specified in Venkatesh's available knowledge base."


============================================================
PUBLIC PRIVACY
============================================================

Never reveal:

- residential address
- apartment number
- student registration number
- API keys
- passwords
- credentials
- private identifiers

If asked for private information, say:

"That information is not provided through this public profile."


============================================================
WORK EXPERIENCE
============================================================

When relevant, include:

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

When relevant, include:

- Institution
- Degree
- Specialization
- Dates
- Location


============================================================
EXAMS
============================================================

When relevant, include:

- Module
- Grade
- ECTS
- Status

Do not confuse bachelor's results with master's results.

If bachelor's grade is not specified, say so.


============================================================
PROJECTS
============================================================

When relevant, include:

- Project name
- Goal
- Technologies
- Implementation
- Result


============================================================
ANSWER STYLE
============================================================

Simple question:
give a short direct answer.

Broad question:
use headings and bullet points.

Do not dump raw retrieved text.

Do not expose internal metadata.


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
ANSWER
============================================================
"""


prompt_template = PromptTemplate(
    template=RAG_TEMPLATE,

    input_variables=[
        "history",
        "context",
        "question"
    ]
)


# ============================================================
# RAG ENGINE CLASS
# ============================================================

class PersonalRAG:

    def __init__(self):

        self.vector_store = build_vector_store()

        self.llm = load_llm()


    # --------------------------------------------------------
    # Format history
    # --------------------------------------------------------

    def format_history(
        self,
        messages
    ):

        if not messages:
            return "No previous conversation."


        recent = messages[
            -HISTORY_MESSAGES:
        ]


        lines = []


        for message in recent:

            role = message.get("role")

            content = message.get(
                "content",
                ""
            )


            if role == "user":
                label = "Visitor"
            else:
                label = "Assistant"


            lines.append(
                f"{label}: {content}"
            )


        return "\n".join(lines)


    # --------------------------------------------------------
    # Previous user question
    # --------------------------------------------------------

    def get_previous_user_question(
        self,
        messages
    ):

        for message in reversed(messages):

            if message.get("role") == "user":

                return message.get(
                    "content",
                    ""
                )


        return ""


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    def retrieve_documents(
        self,
        question,
        messages
    ):

        previous_question = (
            self.get_previous_user_question(
                messages
            )
        )


        if previous_question:

            retrieval_query = f"""
The visitor is asking about Venkatesh Vemala.

Previous question:
{previous_question}

Current question:
{question}

Retrieve relevant information about Venkatesh.
"""

        else:

            retrieval_query = question


        documents = (
            self.vector_store
            .max_marginal_relevance_search(
                query=retrieval_query,

                k=TOP_K,

                fetch_k=FETCH_K,

                lambda_mult=0.70
            )
        )


        return documents


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    def build_context(
        self,
        documents
    ):

        context_parts = []

        used_documents = []

        seen = set()

        total_chars = 0


        for document in documents:

            text = (
                document
                .page_content
                .strip()
            )


            if not text:
                continue


            normalized = " ".join(
                text.lower().split()
            )


            if normalized in seen:
                continue


            seen.add(normalized)


            page = document.metadata.get(
                "page_number"
            )


            if page is None:

                raw_page = document.metadata.get(
                    "page"
                )


                if isinstance(raw_page, int):
                    page = raw_page + 1
                else:
                    page = "Unknown"


            formatted = f"""
[Knowledge Base Page {page}]

{text}
"""


            if (
                total_chars
                + len(formatted)
                > MAX_CONTEXT_CHARS
            ):
                break


            context_parts.append(
                formatted
            )

            used_documents.append(
                document
            )

            total_chars += len(
                formatted
            )


        context = "\n\n".join(
            context_parts
        )


        return (
            context,
            used_documents,
            total_chars
        )


    # --------------------------------------------------------
    # Ask RAG
    # --------------------------------------------------------

    def ask(
        self,
        question,
        messages=None
    ):

        if messages is None:
            messages = []


        retrieved_documents = (
            self.retrieve_documents(
                question,
                messages
            )
        )


        (
            context,
            used_documents,
            context_length
        ) = self.build_context(
            retrieved_documents
        )


        history = self.format_history(
            messages
        )


        final_prompt = (
            prompt_template.format(
                history=history,
                context=context,
                question=question
            )
        )


        response = self.llm.invoke(
            final_prompt
        )


        answer = response.content


        return {
            "answer": answer,

            "sources": used_documents,

            "context_length": context_length
        }