import sounddevice as sd
import numpy as np
import torch
import speech_recognition as sr
from silero_vad import load_silero_vad, get_speech_timestamps
from agents.function.agent import run as afrun

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Load Silero VAD
model = load_silero_vad()
recognizer = sr.Recognizer()

print("🎤 Listening...")

# final_speech = ""

def record_and_transcribe():
    # global final_speech
    # Record audio chunk

    audio = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()

    # Convert to float32 waveform for VAD
    waveform = torch.tensor(audio.T, dtype=torch.float32) / 32768.0

    # Check for speech using Silero VAD
    speech = get_speech_timestamps(waveform, model, sampling_rate=SAMPLE_RATE)

    if not speech:
        print("🤫 Silence.")
        return ""

    # Convert raw PCM to bytes for recognizer
    audio_data = sr.AudioData(audio.tobytes(), SAMPLE_RATE, 2)  # 2 bytes = 16-bit PCM
    print(audio_data)
    try:
        text = recognizer.recognize_google(audio_data)
        print(f"🗣️ {text}")
        return text
        # final_speech += text

    except sr.UnknownValueError:
        print("⚠️ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"🔌 Request error: {e}")
        return ""


def run():
    # Main loop
    try:
        wait = 0
        final_speech = ""
        while True:
            if wait > 5:
                afrun(final_speech)
                break
            else:
                final_speech += record_and_transcribe()
            wait += 1

    except KeyboardInterrupt:
        print("🛑 Stopped.")
