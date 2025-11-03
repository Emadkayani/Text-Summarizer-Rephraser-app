import streamlit as st
from transformers import pipeline

# ------------------------------
# Load models
# ------------------------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    return summarizer, paraphraser

summarizer, paraphraser = load_models()

# ------------------------------
# Define functions
# ------------------------------
def chunk_text(text, chunk_size=900):
    """Split long text into chunks."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

def summarize_text(text, max_length=300, min_length=80):
    """Summarize long or short text adaptively."""
    if not text.strip():
        return "⚠️ Please enter some text to summarize."

    input_length = len(text.split())
    if input_length < 60:
        max_length = max(30, int(input_length * 0.7))
        min_length = max(10, int(input_length * 0.3))
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']

    summaries = []
    for chunk in chunk_text(text):
        result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(result[0]['summary_text'])
    return " ".join(summaries)

def rephrase_text(text, num_return_sequences=3):
    """Generate diverse rephrased outputs."""
    if not text.strip():
        return ["⚠️ Please enter text to rephrase."]
    prompt = f"paraphrase: {text}"
    outputs = paraphraser(
        prompt,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        num_beams=1,
        max_length=256
    )
    return [o['generated_text'] for o in outputs]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Text Summarizer & Rephraser", layout="centered")
st.title("🧠 Text Summarizer & Rephraser App")
st.write("Enter your text below to summarize or rephrase it using AI models.")

user_input = st.text_area("✍️ Enter Text", height=200)

col1, col2 = st.columns(2)
with col1:
    summarize_btn = st.button("Summarize 📝")
with col2:
    rephrase_btn = st.button("Rephrase 🔁")

if summarize_btn:
    with st.spinner("Generating summary..."):
        summary = summarize_text(user_input)
    st.subheader("🧾 Summary:")
    st.write(summary)

elif rephrase_btn:
    with st.spinner("Generating rephrased versions..."):
        rephrased = rephrase_text(user_input)
    st.subheader("🔁 Rephrased Versions:")
    for i, p in enumerate(rephrased, 1):
        st.write(f"**Version {i}:** {p}")
