import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset
from gtts import gTTS
import base64
import os
from pathlib import Path
import io

# Page config
st.set_page_config(page_title="Blender Chat Bot", layout="wide")

# シンプルな音声再生用のHTML関数
def create_audio_player(audio_data):
    b64 = base64.b64encode(audio_data).decode()
    
    md = f"""
        <audio autoplay controls style="width: 100%">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    return st.markdown(md, unsafe_allow_html=True)

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

# 音声を生成する関数
def generate_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
st.title("Blender Chat Bot")

# サイドバー設定
with st.sidebar:
    enable_tts = st.checkbox("Enable Text-to-Speech", value=True)
    lang = st.selectbox("Language", ['en'], index=0)

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
        
        # 音声読み上げが有効な場合、応答を音声に変換して再生
        if enable_tts:
            with st.spinner("Generating..."):
                audio_data = generate_speech(response, lang)
                if audio_data:
                    create_audio_player(audio_data)
