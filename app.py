import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "prithivida/parrot_paraphraser_on_T5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="AI Paraphraser", page_icon="✨", layout="centered")

st.title("✨ AI Paraphraser")
st.write("Generate multiple creative rephrasings of your sentence using a T5-based AI model.")
st.markdown("---")

input_text = st.text_area("✍️ Enter text to rephrase:", height=150, placeholder="Type or paste a sentence or paragraph...")

if st.button("🔄 Rephrase"):
    if input_text.strip():
        with st.spinner("Rephrasing your text... please wait ⏳"):
            # Prepare model input correctly (no long instruction)
            text = "paraphrase: " + input_text

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            # Generate diverse paraphrases
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                num_return_sequences=5,
                do_sample=True,
                temperature=1.3,
                top_k=120,
                top_p=0.95,
                repetition_penalty=2.5,
                early_stopping=True
            )

            decoded = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]

            # Filter duplicates and irrelevant outputs
            rephrased = []
            for d in decoded:
                if d and d.lower() != input_text.lower() and d not in rephrased:
                    rephrased.append(d)
                if len(rephrased) >= 3:
                    break

        st.success("✅ Rephrasing complete!")
        st.markdown("### ✨ Rephrased Versions:")
        if rephrased:
            for i, version in enumerate(rephrased, 1):
                st.markdown(f"**Version {i}:** {version}")
        else:
            st.warning("⚠️ The model couldn't generate distinct rephrasings. Try rephrasing a slightly longer text.")
    else:
        st.warning("⚠️ Please enter some text first.")


