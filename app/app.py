import gradio as gr
import os
import re
import torch
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

# =========================
# Model
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tokenizer_path = os.path.join(BASE_DIR, "saved_files", "tokenizer")
model_path = os.path.join(BASE_DIR, "saved_files", "summarization_model")

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================
# Clean text
# =========================
def clean_text(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()

    return text

# =========================
# Chat function (Streaming)
# =========================
def chat(message, history):

    message = clean_text(message)

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    yield "Bare with me..."

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=175,
            num_beams=4,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    streamed = ""
    for word in response.split():
        streamed += word + " "
        time.sleep(0.02)
        yield streamed.strip()

# =========================
# UI
# =========================
css = """
.gradio-container {
    background: radial-gradient(circle at top, #111827, #0b1220);
    color: white;
}

footer {display:none !important;}
"""

theme = gr.themes.Soft(primary_hue="blue", radius_size="lg")

with gr.Blocks(theme=theme, css=css) as demo:

    gr.Markdown("# Dialogue Summarizer")

    # ================= CHAT =================
    chatbot = gr.ChatInterface(
        fn=chat,
        examples=None
    )

demo.queue().launch()
