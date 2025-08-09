# speech
# 🎙️ English → Telugu Voice Assistant

This project is a **speech processing tool** that:

1. Takes an **English sentence** (typed in by you).
2. Reads it out loud in **English**.
3. Listens to that audio and **transcribes** it back into text.
4. **Translates** the English text into **Telugu**.
5. Reads the Telugu translation out loud.
6. Shows each step of the process in a live log.

---

## 📦 Requirements

Before running the app, make sure you have:

- **Python 3.9+**
- **pip** (Python package installer)
- **Internet connection** (models are downloaded the first time)
- **FFmpeg** installed on your system (needed for audio processing)

pip install -r requirements.txt to download these

---

## 📥 Installation

1. Clone the repository (or download ZIP):

git clone https://github.com/bavesh-aispace/speech
cd speech


2.Running the App:
Make sure you are in the project folder.

Run: python app.py

Open your browser and go to: http://127.0.0.1:5000



## 💡 How It Works:
You type an English sentence.

English TTS → Uses Microsoft Edge TTS to generate English speech.

Speech Recognition → Uses Whisper ASR to transcribe English audio into text.

Translation → Uses IndicTrans2 to translate text into Telugu.

Telugu TTS → Uses Facebook MMS-TTS to generate Telugu audio.

All results and audio files are shown on the webpage.




## 🖼 Features:
Live log updates using Server-Sent Events (SSE)

Two audio players (English & Telugu output)

Real-time transcription and translation

Simple and clean web interface

