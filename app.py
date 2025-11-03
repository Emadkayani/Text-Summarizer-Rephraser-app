import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ——————————————————————————————
# Page config (must be first)
# ——————————————————————————————
st.set_page_config(
    page_title="AI Text Rephraser",
    page_icon="🪄",
    layout="centered",
)

# ——————————————————————————————
# Model loading (cached)
# ——————————————————————————————
@st.cache_resource
def load_model(model_name: str = "humarin/chatgpt_paraphraser_on_T5_base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ——————————————————————————————
# UI Header
# ——————————————————————————————
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>🪄 AI Text Rephraser</h1>
    <p style='text-align: center; color: gray;'>
    Generate multiple creative rewrites of your text while preserving meaning.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ——————————————————————————————
# User Input
# ——————————————————————————————
input_text = st.text_area(
    "✍️ Enter text to rephrase:",
    height=150,
    placeholder="Type or paste your sentence or paragraph here..."
)

# ——————————————————————————————
# Generation logic
# ——————————————————————————————
def generate_rephrases(
    text: str,
    num_return_sequences: int = 5,
    max_length: int = 128,
    temperature: float = 1.2,
    top_k: int = 100,
    top_p: float = 0.95,
    repetition_penalty: float = 2.5,
) -> list[str]:
    prompt = f"paraphrase: {text}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        early_stopping=True,
        num_beams=1
    )

    decoded = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
    unique = []
    for d in decoded:
        if d and d.lower() != text.lower() and d.lower() not in (u.lower() for u in unique):
            unique.append(d)
        if len(unique) >= 3:
            break
    return unique

# ——————————————————————————————
# State Management
# ——————————————————————————————
if "rephrases" not in st.session_state:
    st.session_state.rephrases = []

# ——————————————————————————————
# Main UI
# ——————————————————————————————
if st.button("🔄 Rephrase"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text first.")
    elif len(input_text.split()) < 6:
        st.warning("⚠️ Please enter a longer text (at least 6 words) for meaningful rephrasing.")
    else:
        with st.spinner("🌀 Generating rewrites..."):
            st.session_state.rephrases = generate_rephrases(input_text)
        if not st.session_state.rephrases:
            st.error("❗ Could not generate distinct rewrites. Try a different sentence or longer text.")
        else:
            st.success("✅ Rephrasing complete!")

# ——————————————————————————————
# Display Results with Copy Buttons
# ——————————————————————————————
if st.session_state.rephrases:
    st.markdown("### 🔹 Rephrased Versions:")
    for i, version in enumerate(st.session_state.rephrases, start=1):
        safe_version = version.replace("`", "'").replace('"', "'")  # avoid quote issues
        st.markdown(
            f"""
            <div style='background-color:#F8F9FA; border-radius:8px; padding:12px; margin-top:8px;'>
                <b>Version {i}:</b> {safe_version}
                <br>
                <button 
                    onclick="navigator.clipboard.writeText(`{safe_version}`); 
                             alert('✅ Version {i} copied to clipboard!');"
                    style="margin-top:8px; background-color:#4A90E2; color:white; border:none;
                           border-radius:6px; padding:6px 12px; cursor:pointer;">
                    📋 Copy Version {i}
                </button>
            </div>
            """,
            unsafe_allow_html=True,
        )

