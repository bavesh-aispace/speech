import asyncio
import os
import torch
import torchaudio
import soundfile as sf
from flask import Flask, render_template, request, Response, send_from_directory
from dotenv import load_dotenv
import edge_tts
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModelForTextToWaveform,
    AutoModelForSequenceClassification
)
from IndicTransToolkit.processor import IndicProcessor
import torch.nn.functional as F
import time

load_dotenv()
app = Flask(__name__)

# ===================== Sentence Type Detection Setup =====================
MODEL_NAME = "textattack/bert-base-uncased-SST-2"  # Replace with your fine-tuned model
SENTENCE_TYPES = ["declarative", "interrogative", "imperative", "exclamatory"]

sent_type_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sent_type_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sent_type_model.to(device)
sent_type_model.eval()

def predict_sentence_type(text):
    inputs = sent_type_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = sent_type_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred_id = probs.argmax(dim=1).item()
    return SENTENCE_TYPES[pred_id % len(SENTENCE_TYPES)]

def heuristic_sentence_type(sentence):
    sentence = sentence.strip().lower()
    question_starts = (
        "who", "what", "when", "where", "why", "how",
        "is", "are", "can", "do", "does", "did", "will", "would", "could", "should"
    )
    imperative_starts = ("please", "kindly", "do", "let's", "go", "come", "stop", "start", "take", "make")
    if sentence.endswith("!"):
        return "exclamatory"
    if sentence.startswith(question_starts):
        return "interrogative"
    if sentence.startswith(imperative_starts):
        return "imperative"
    return "declarative"

def detect_sentence_type(sentence):
    # Use heuristic for now; switch to ML by uncommenting below:
    # return predict_sentence_type(sentence)
    return heuristic_sentence_type(sentence)

# ===================== Your Existing Model Loads =====================
model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")
processor = IndicProcessor(inference=True)

# Load ASR pipeline once
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=0 if torch.cuda.is_available() else -1,
    generate_kwargs={"language": "en"}
)

# Load Telugu TTS model once
telugu_tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tel")
telugu_tts_model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-tel")

# ===================== Your Existing Pipeline Functions =====================
async def generate_english_speech(text, log):
    sentence_type = detect_sentence_type(text)
    await log(f"[INFO] Text: \"{text}\"")
    await log(f"[INFO] Type: {sentence_type.capitalize()}")
    output_file = "static/english_input.wav"
    communicate = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
    await communicate.save(output_file)
    await log(f"[INFO] English audio saved as {output_file}")

def transcribe(audio_path, log_func):
    log_func("[INFO] Transcribing English audio...")
    audio, sr = sf.read(audio_path)
    result = asr_pipeline({"array": audio, "sampling_rate": sr})
    return result['text']

def translate_to_telugu(text, log_func):
    log_func("[INFO] Translating to Telugu...")
    src_lang, tgt_lang = "eng_Latn", "tel_Telu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_batch = processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(processed_batch, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=5, use_cache=False)
    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return processor.postprocess_batch(translated, lang=tgt_lang)[0]

def synthesize_telugu(text, output_file, log_func):
    log_func("[INFO] Synthesizing Telugu speech...")
    inputs = telugu_tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = telugu_tts_model(**inputs)
    waveform = output.waveform.squeeze().cpu()
    torchaudio.save(output_file, waveform.unsqueeze(0), 16000)
    log_func(f"[INFO] Telugu audio saved as {output_file}")

async def async_log_generator(queue):
    while True:
        msg = await queue.get()
        if msg is None:
            break
        yield f"data:{msg}\n\n"

# ===================== Flask Routes =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_pipeline():
    user_text = request.form["user_text"]

    def event_stream():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        queue = asyncio.Queue()

        async def async_generate_wrapper():
            async def log(msg):
                await queue.put(msg)

            # 1. English TTS
            await generate_english_speech(user_text, log)

            # 2. Transcription
            await log("[INFO] Starting transcription...")
            eng_text = transcribe("static/english_input.wav", lambda m: loop.create_task(log(m)))
            await log(f"[INFO] Transcription result: {eng_text}")

            # 3. Translation
            tel_text = translate_to_telugu(eng_text, lambda m: loop.create_task(log(m)))
            await log(f"[INFO] Translation result: {tel_text}")

            # 4. Telugu TTS
            synthesize_telugu(tel_text, "static/telugu_output.wav", lambda m: loop.create_task(log(m)))
            await log("[INFO] Telugu speech synthesis done.")

            await queue.put(f"__RESULT__::{eng_text}::{tel_text}::/static/telugu_output.wav?ts={int(time.time())}")
            await queue.put(None)

        loop.create_task(async_generate_wrapper())

        while True:
            msg = loop.run_until_complete(queue.get())
            if msg is None:
                break
            yield f"data:{msg}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

# ===================== Run Server =====================
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
