"""
Grameenphone FAQ Chatbot - Streamlit interface.
Run: uv run streamlit run app.py  (after training with train.py)
"""

import json
import torch
import streamlit as st
from model import IntentClassifier, encode_text
from retriever import load_faq_kb, search_faq


@st.cache_resource
def load_model():
    with open("models/config.json") as f:
        config = json.load(f)
    with open("models/word2idx.json") as f:
        word2idx = json.load(f)
    with open("models/idx2label.json") as f:
        idx2label = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentClassifier(len(word2idx), config["embed_dim"], config["hidden_dim"], config["num_classes"])
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device, weights_only=True))
    model.eval()
    return model, word2idx, idx2label, config, device


model, word2idx, idx2label, config, device = load_model()
knowledge_base = load_faq_kb()


def predict_intent(text):
    """Predict intent label and confidence from user text."""
    ids = encode_text(text, word2idx, config["max_len"])
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()
    return idx2label[str(pred)], probs[0][pred].item()


# --- Streamlit UI ---

st.set_page_config(page_title="GP FAQ Chatbot", page_icon="📱")
st.title("📱 Grameenphone Customer Service Chatbot")

GREETING = (
    "Hello! Welcome to Grameenphone Customer Service. "
    "How can I help you today? You can ask me about balance, recharge, "
    "data packages, SIM replacement, and more."
)
LOW_CONFIDENCE_MSG = (
    "Sorry, I couldn't understand your question. "
    "Could you please rephrase it? You can also call 121 for direct support."
)
CONFIDENCE_THRESHOLD = 0.50

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question (English or Bangla)..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    intent, confidence = predict_intent(query)

    if confidence < CONFIDENCE_THRESHOLD:
        response = LOW_CONFIDENCE_MSG
    else:
        answer = search_faq(query, knowledge_base, intent=intent)
        response = f"{answer}\n\n*Intent: `{intent}` | Confidence: {confidence:.0%}*"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
