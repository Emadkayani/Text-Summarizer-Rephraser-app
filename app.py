import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Rephraser", layout="centered")

# ---- TITLE ----
st.title("✨ AI Text Rephraser")

@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "Vamsi/T5_Paraphrase_Paws"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---- INPUT ----
text_input = st.text_area("✏️ Enter text to rephrase:", height=150, placeholder="Type or paste text here...")

if st.button("Generate Rephrased Versions"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("⚙️ Generating rewrites... please wait (10–20s)"):
            try:
                # Prepare input
                input_text = f"paraphrase: {text_input} </s>"
                encoding = tokenizer(
                    input_text,
                    padding="longest",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt"
                )

                # Generate diverse outputs
                outputs = model.generate(
                    **encoding,
                    max_length=256,
                    num_return_sequences=3,
                    num_beams=5,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    early_stopping=True
                )

                st.success("✅ Rewrites generated successfully!")

                # Display all versions
                for i, output in enumerate(outputs):
                    rephrased = tokenizer.decode(output, skip_special_tokens=True)
                    st.markdown(f"**Version {i+1}:** {rephrased}")

            except torch.cuda.OutOfMemoryError:
                st.error("💥 Out of memory! Try shorter input text.")
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
