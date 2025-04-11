
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned model from Hugging Face
MODEL_NAME = "venkatasaig/venkatasai_flan_t5_base"  # replace with your actual HF repo name

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Title
st.title("Fine-Tuned Flan-T5 QA Assistant")

# Text input
user_input = st.text_area("Enter your instruction (e.g., 'Show GTIN and product code for...')")

# Generate response
if st.button("Generate Answer") and user_input.strip():
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_new_tokens=256)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    st.markdown("### 🧾 Response:")
    st.success(output_text)
