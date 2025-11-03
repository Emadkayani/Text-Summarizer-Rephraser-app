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
st.write("Generate multiple creative rephrasings for your sentence using a T5-based model.")

st.markdown("---")

input_text = st.text_area("✍️ Enter your text here:", height=150, placeholder="Type a sentence or paragraph...")

if st.button("🔄 Rephrase"):
    if input_text.strip():
        with st.spinner("Rephrasing your text... please wait ⏳"):
            # Strong prompt for creativity
            prompt = (
                f"Paraphrase the following sentence in 5 different creative ways:\n"
                f"\"{input_text}\"\n\n"
                "Each paraphrase should use different vocabulary and sentence structure while keeping the same meaning."
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=512
            ).to(model.device)

            # Generate multiple paraphrases
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_return_sequences=10,   # generate extra for variety
                do_sample=True,
                temperature=1.3,           # increase randomness
                top_k=100,
                top_p=0.9,
                repetition_penalty=3.0,
                diversity_penalty=1.5,
                early_stopping=True
            )

            # Decode and remove duplicates
            decoded = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
            rephrased_versions = []
            for d in decoded:
                if d and d.lower() not in [r.lower() for r in rephrased_versions]:
                    rephrased_versions.append(d)
                if len(rephrased_versions) >= 3:
                    break

        st.success("✅ Paraphrasing complete!")

        st.markdown("### ✨ Rephrased Versions:")
        for i, version in enumerate(rephrased_versions, 1):
            st.markdown(f"**Version {i}:** {version}")
    else:
        st.warning("⚠️ Please enter some text first.")

st.markdown("---")

