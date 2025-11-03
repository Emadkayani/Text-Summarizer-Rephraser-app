# 🪄 AI Text Rephraser (Streamlit App)

An interactive web app that rephrases any text into multiple creative variations while preserving meaning — built with **Streamlit** and **Hugging Face Transformers**.

---

## 🚀 Features
- Generates multiple paraphrased versions of input text  
- Uses a fine-tuned **T5** paraphrasing model (`humarin/chatgpt_paraphraser_on_T5_base`)  
- Clean, minimal Streamlit interface  
- **Copy to clipboard** for each generated version (instant, no rerun)  
- Works seamlessly on **Streamlit Cloud** or **Hugging Face Spaces**

---

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **Framework:** Streamlit 1.51+
- **Model Library:** Transformers 4.46+
- **Backend:** CPU (PyTorch 2.9)
- **Frontend:** HTML, JS (via `st.components.html`)

---

## 🧩 Installation (Local)
1. Clone the repository:
   ```bash
   git clone https://github.com/Emadkayani/text-rephraser-app.git
   cd text-rephraser-app
