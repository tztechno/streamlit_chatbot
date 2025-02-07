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

# モバイル対応の強化された音声再生用のHTML関数
def create_audio_player(audio_data):
    b64 = base64.b64encode(audio_data).decode()
    
    # より積極的な自動再生の試行
    md = f"""
        <div id="audio-container">
            <audio id="audio-player" autoplay playsinline muted controls style="width: 100%">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>

            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    const audioPlayer = document.getElementById('audio-player');
                    
                    // ミュートを解除して再生を試みる
                    const playAudio = async () => {{
                        try {{
                            audioPlayer.muted = false;
                            await audioPlayer.play();
                        }} catch (error) {{
                            console.log('Autoplay failed:', error);
                            // エラー時は手動再生用のコントロールを表示したまま
                            audioPlayer.controls = true;
                        }}
                    }};

                    // タッチイベントとクリックイベントの両方で再生を試みる
                    const startPlayback = () => {{
                        playAudio();
                        // イベントリスナーを削除（一度だけ実行）
                        document.removeEventListener('touchstart', startPlayback);
                        document.removeEventListener('click', startPlayback);
                    }};

                    // 様々なイベントで再生を試みる
                    document.addEventListener('touchstart', startPlayback, {{once: true}});
                    document.addEventListener('click', startPlayback, {{once: true}});
                    
                    // 直接再生も試みる
                    playAudio();
                }});
            </script>
        </div>
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

# 音声を生成する関数（英語固定）
def generate_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        st.error(f"音声生成エラー: {str(e)}")
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
st.title("Blender Chat Bot")

# サイドバー設定（シンプル化）
with st.sidebar:
    enable_tts = st.checkbox("Enable Text-to-Speech", value=True)

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
            with st.spinner("Generating audio..."):
                audio_data = generate_speech(response)
                if audio_data:
                    create_audio_player(audio_data)


