import customtkinter as ctk

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


# -------------------------------------------------
# Ollama model
# -------------------------------------------------

MODEL_NAME = "llama3.2"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.3
)


# -------------------------------------------------
# Conversation memory
# -------------------------------------------------

messages = []


# -------------------------------------------------
# App
# -------------------------------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()

app.title("Local Ollama AI")
app.geometry("1000x700")


# -------------------------------------------------
# Title
# -------------------------------------------------

title = ctk.CTkLabel(
    app,
    text="Local Ollama AI",
    font=("Arial", 28, "bold")
)

title.pack(pady=(20, 5))


subtitle = ctk.CTkLabel(
    app,
    text=f"Running locally with {MODEL_NAME}",
    font=("Arial", 14)
)

subtitle.pack(pady=(0, 15))


# -------------------------------------------------
# Chat display
# -------------------------------------------------

chat_box = ctk.CTkTextbox(
    app,
    width=900,
    height=500,
    font=("Arial", 15),
    wrap="word"
)

chat_box.pack(
    padx=30,
    pady=10,
    fill="both",
    expand=True
)

chat_box.configure(state="disabled")


# -------------------------------------------------
# Functions
# -------------------------------------------------

def add_message(sender, text):

    chat_box.configure(state="normal")

    chat_box.insert(
        "end",
        f"\n{sender}:\n{text}\n"
    )

    chat_box.configure(state="disabled")

    chat_box.see("end")


def send_message():

    user_text = input_box.get().strip()

    if not user_text:
        return

    input_box.delete(0, "end")

    add_message(
        "You",
        user_text
    )

    messages.append(
        HumanMessage(
            content=user_text
        )
    )

    try:

        response = llm.invoke(
            messages
        )

        answer = response.content

        add_message(
            "AI",
            answer
        )

        messages.append(
            AIMessage(
                content=answer
            )
        )

    except Exception as e:

        add_message(
            "Error",
            str(e)
        )


def clear_chat():

    messages.clear()

    chat_box.configure(state="normal")

    chat_box.delete(
        "1.0",
        "end"
    )

    chat_box.configure(state="disabled")


# -------------------------------------------------
# Bottom input area
# -------------------------------------------------

bottom_frame = ctk.CTkFrame(app)

bottom_frame.pack(
    padx=30,
    pady=20,
    fill="x"
)


input_box = ctk.CTkEntry(
    bottom_frame,
    placeholder_text="Ask your local AI...",
    font=("Arial", 15)
)

input_box.pack(
    side="left",
    padx=10,
    pady=10,
    fill="x",
    expand=True
)


send_button = ctk.CTkButton(
    bottom_frame,
    text="Send",
    command=send_message,
    width=100
)

send_button.pack(
    side="left",
    padx=5
)


clear_button = ctk.CTkButton(
    bottom_frame,
    text="Clear",
    command=clear_chat,
    width=100
)

clear_button.pack(
    side="left",
    padx=5
)


# Press Enter to send
input_box.bind(
    "<Return>",
    lambda event: send_message()
)


# -------------------------------------------------
# Start app
# -------------------------------------------------

app.mainloop()