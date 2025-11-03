import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI Text Rephraser",
    page_icon="🪄",
    layout="centered",
)

# -------------------------------------------------
# App Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>🪄 AI Text Rephraser</h1>
    <p style='text-align: center; color: gray;'>
    Instantly generate unique, natural rephrasings of any text.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Load Model (cached for performance)
# -------------------------------------------------
@st.cache_resource
def load_model():
    # 🔹 Choose a model: (Uncomment one)
    # model_name = "ramsrigouthamg/t5_paraphraser"  # fast, small, basic
    model_name = "Vamsi/T5_Paraphrase_Paws"        # better quality and diversity

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -------------------------------------------------
# User Input
# -------------------------------------------------
st.markdown("### ✍️ Enter text to rephrase:")
input_text = st.text_area("", placeholder="Type or paste your text here...", height=180)

# -------------------------------------------------
# Rephrasing Logic
# -------------------------------------------------
if st.button("✨ Generate Rephrased Versions"):
    if input_text.strip():
        if len(input_text.split()) < 5:
            st.warning("⚠️ Please enter a longer sentence (at least 5 words).")
        else:
            with st.spinner("🔄 Rephrasing your text... please wait"):
                prompt = (
                    f"paraphrase this in different ways: {input_text}. "
                    "Each version should use new words or phrasing but keep the same meaning."
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=256
                ).to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_return_sequences=5,
                    do_sample=True,
                    temperature=1.1,
                    top_k=80,
                    top_p=0.92,
                    repetition_penalty=2.5,
                    early_stopping=True
                )

                # Decode & remove duplicates
                rephrased_versions = list({
                    tokenizer.decode(output, skip_special_tokens=True).strip()
                    for output in outputs
                })

            st.success("✅ Rephrasing complete!")

            if not rephrased_versions:
                st.warning("Model couldn't produce diverse outputs. Try rephrasing a longer sentence.")
            else:
                st.markdown("### 🔹 Rephrased Versions:")
                for i, text in enumerate(rephrased_versions, 1):
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="
                                background-color:#F8F9FA;
                                border-radius:10px;
                                padding:15px;
                                margin-top:10px;
                                border-left:5px solid #4A90E2;">
                                <b>Version {i}:</b><br>{text}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.button(f"📋 Copy Version {i}", key=f"copy_{i}")
    else:
        st.warning("Please enter some text to rephrase!")

