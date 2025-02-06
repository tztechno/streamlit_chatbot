import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset

# Page config
st.set_page_config(page_title="Chat Bot", layout="wide")

# Initialize model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Chat handling functions
def tokenize_data(inputs, max_length=64):
    return tokenizer(
        list(inputs), 
        max_length=max_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }

def generate_response(input_text):
    inputs_enc = tokenize_data([input_text])
    input_ids = inputs_enc["input_ids"].to(device)
    attention_mask = inputs_enc["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            repetition_penalty=1.2,
            max_length=128
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
st.title("Chat Bot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
