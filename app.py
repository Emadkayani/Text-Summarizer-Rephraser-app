import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Must be the first Streamlit command
st.set_page_config(page_title="AI Text Rephraser", layout="centered")

st.title("🪄 AI Text Rephraser")
st.write("Enter any text below and get rephrased versions instantly!")

@st.cache_resource
def load_model():
    """Load lightweight T5 paraphrasing model."""
    model_name = "ramsrigouthamg/t5_paraphraser"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# User input
input_text = st.text_area("✍️ Enter your text:", height=200)

if st.button("Generate Rephrased Versions"):
    if input_text.strip():
        with st.spinner("Rephrasing in progress..."):
            # Handle short text intelligently
            if len(input_text.split()) < 5:
                st.warning("⚠️ Please enter a longer sentence (at least 5 words).")
            else:
                inputs = tokenizer(
                    f"paraphrase: {input_text}",
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=256
                ).to(model.device)

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

        st.markdown("### 🔹 Rephrased Versions:")
        for i, text in enumerate(rephrased_versions, 1):
            st.write(f"**{i}.** {text}")
    else:
        st.warning("Please enter some text to rephrase!")
