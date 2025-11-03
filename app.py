import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 🔹 MUST BE FIRST Streamlit command
st.set_page_config(page_title="Text Summarizer & Rephraser", layout="centered")

# 🔹 App title
st.title("📝 Text Summarizer & Rephraser")

# Load models (cached for performance)
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    rephraser_model_name = "Vamsi/T5_Paraphrase_Paws"
    tokenizer = AutoTokenizer.from_pretrained(rephraser_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(rephraser_model_name)
    return summarizer, tokenizer, model

summarizer, tokenizer, model = load_models()

# 🔹 Text input
input_text = st.text_area("Enter your text here:", height=200)

if st.button("Generate Summary and Rephrased Versions"):
    if input_text.strip():
        with st.spinner("Processing... ⏳"):
            # Summarization
            summary = summarizer(
                input_text,
                max_length=200,
                min_length=50,
                do_sample=False
            )[0]['summary_text']

            # Rephrasing with randomness
            inputs = tokenizer(
                f"paraphrase: {summary}",
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding="longest"
            )

            outputs = model.generate(
                **inputs,
                max_length=256,
                num_return_sequences=3,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95
            )

            rephrased_versions = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

        # 🔹 Display results
        st.markdown("### 🔹 Summary:")
        st.write(summary)

        st.markdown("### 🔹 Rephrased Versions:")
        for i, text in enumerate(rephrased_versions, start=1):
            st.write(f"**{i}.** {text}")
    else:
        st.warning("Please enter some text to summarize and rephrase.")
