import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import streamlit.components.v1 as components

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
# Header
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
# Session state
# ——————————————————————————————
if "rephrases" not in st.session_state:
    st.session_state.rephrases = []

# ——————————————————————————————
# Button
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
# Display results with working copy buttons
# ——————————————————————————————
if st.session_state.rephrases:
    st.markdown("### 🔹 Rephrased Versions:")

    # Encode safely for HTML/JS
    rephrases_json = json.dumps(st.session_state.rephrases)

    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <style>
        .card {{
          background: #F8F9FA;
          border-radius: 8px;
          padding: 12px;
          margin-top: 8px;
          border-left: 5px solid #4A90E2;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        }}
        .copy-btn {{
          margin-top:8px;
          background-color:#4A90E2;
          color:white;
          border:none;
          border-radius:6px;
          padding:6px 12px;
          cursor:pointer;
        }}
        .toast {{
          visibility: hidden;
          min-width: 200px;
          margin-left: -100px;
          background-color: #333;
          color: #fff;
          text-align: center;
          border-radius: 4px;
          padding: 8px;
          position: fixed;
          z-index: 9999;
          left: 50%;
          bottom: 30px;
          font-size: 14px;
        }}
        .toast.show {{
          visibility: visible;
          -webkit-animation: fadein 0.3s, fadeout 0.6s 1.2s;
          animation: fadein 0.3s, fadeout 0.6s 1.2s;
        }}
        @-webkit-keyframes fadein {{ from {{bottom: 0; opacity: 0;}} to {{bottom: 30px; opacity: 1;}} }}
        @keyframes fadein {{ from {{bottom: 0; opacity: 0;}} to {{bottom: 30px; opacity: 1;}} }}
        @-webkit-keyframes fadeout {{ from {{bottom: 30px; opacity: 1;}} to {{bottom: 0; opacity: 0;}} }}
        @keyframes fadeout {{ from {{bottom: 30px; opacity: 1;}} to {{bottom: 0; opacity: 0;}} }}
      </style>
    </head>
    <body>
      <div id="container"></div>
      <div id="toast" class="toast">Copied to clipboard</div>

      <script>
        (function() {{
          const rephrases = {rephrases_json};
          const container = document.getElementById('container');

          function showToast(msg) {{
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.className = 'toast show';
            setTimeout(() => {{ toast.className = 'toast'; }}, 1800);
          }}

          function copyText(text) {{
            if (navigator.clipboard && navigator.clipboard.writeText) {{
              navigator.clipboard.writeText(text).then(() => {{
                showToast('✅ Copied to clipboard');
              }}, (err) => {{
                fallbackCopy(text);
              }});
            }} else {{
              fallbackCopy(text);
            }}
          }}

          function fallbackCopy(text) {{
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            try {{
              document.execCommand('copy');
              showToast('✅ Copied to clipboard');
            }} catch (err) {{
              showToast('⚠️ Copy failed');
            }}
            document.body.removeChild(ta);
          }}

          rephrases.forEach((r, idx) => {{
            const card = document.createElement('div');
            card.className = 'card';
            const title = document.createElement('div');
            title.innerHTML = '<b>Version ' + (idx + 1) + ':</b>';
            const para = document.createElement('div');
            para.style.marginTop = '6px';
            para.textContent = r;

            const btn = document.createElement('button');
            btn.className = 'copy-btn';
            btn.innerHTML = '📋 Copy Version ' + (idx + 1);
            btn.onclick = function() {{
              copyText(r);
            }};

            card.appendChild(title);
            card.appendChild(para);
            card.appendChild(btn);
            container.appendChild(card);
          }});
        }})();
      </script>
    </body>
    </html>
    """

    components.html(html, height=200 + 140 * len(st.session_state.rephrases), scrolling=True)


