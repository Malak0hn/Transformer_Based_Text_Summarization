# Transformer-Based Text Summarization

An abstractive text summarization project using a fine-tuned **T5 (Text-to-Text Transfer Transformer)** model to generate concise summaries from longer text inputs.

## Overview

This project implements a sequence-to-sequence approach to abstractive text summarization. The model generates new text that captures the main ideas of the input rather than simply extracting existing sentences.

The trained model and tokenizer are saved and integrated into a Streamlit application, providing an interactive interface for generating summaries.

## Model

The project uses a fine-tuned T5 model for conditional text generation.

The inference pipeline consists of:
- Text cleaning and preprocessing.
- Tokenization using the saved T5 tokenizer.
- Input truncation to a maximum of 512 tokens.
- Summary generation using the trained model.
- Decoding the generated tokens into readable text.

The model runs on GPU when CUDA is available, with CPU support as a fallback.

## Application

The application provides a simple conversational interface for interacting with the summarization model.

**How to use:**
1. Open the [Dialogue Summarizer](https://dialogue-summarizer-m.streamlit.app/).
2. Enter or paste the text you want to summarize.
3. Submit your input through the chat interface.
4. Wait for the model to generate and display the summary.

## Technologies

Python · PyTorch · Hugging Face Transformers · T5 · SentencePiece · Streamlit

## Project Structure

```text
├── app/          # Application and inference logic
├── data/         # Dataset files
├── notebook/     # Model development and experiments
├── saved_files/  # Trained model and tokenizer
└── requirements.txt
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Malak0hn/Transformer_Based_Text_Summarization.git
cd Transformer_Based_Text_Summarization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app/app.py
```

## Author

**Malak Hussein**
