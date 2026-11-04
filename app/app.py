import streamlit as st
import os
import re
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Dialogue Summarizer",
    page_icon="📝",
    layout="centered"
)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

tokenizer_path = os.path.join(
    BASE_DIR, "saved_files", "tokenizer"
)

model_path = os.path.join(
    BASE_DIR, "saved_files", "summarization_model"
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

    model = T5ForConditionalGeneration.from_pretrained(model_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


try:
    tokenizer, model, device = load_model()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# =========================
# Clean Text
# =========================
def clean_text(text):

    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()

    return text


# =========================
# Summarization
# =========================
def summarize(message):

    message = clean_text(message)

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_length=175,
            num_beams=4,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


# =========================
# Custom CSS
# =========================
st.markdown("""
<style>

.stApp {
    background: radial-gradient(
        circle at top,
        #111827,
        #0b1220
    );
    color: white;
}

h1, p, label {
    color: white;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================
# UI
# =========================
st.title("Dialogue Summarizer")

st.caption("AI-powered text summarization using T5")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
if prompt := st.chat_input("Enter text to summarize..."):

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Generating summary..."):

            try:
                response = summarize(prompt)

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error(f"Summarization error: {e}")
