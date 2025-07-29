import sounddevice as sd
import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

# Load Silero VAD model
model = load_silero_vad()

# Constants
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    # Convert input to float32 tensor for Silero
    mono_audio = torch.from_numpy(indata.copy()).float().reshape(1, -1)

    # Run VAD
    timestamps = get_speech_timestamps(mono_audio, model, sampling_rate=SAMPLE_RATE, threshold=0.5)

    if timestamps:
        print("🗣️ Speaking detected!")
    else:
        print("🤫 Silence.")

def run():
    # Buffer to hold audio chunks
    print("🎤 Starting real-time VAD... Press Ctrl+C to stop.")

    try:
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, callback=audio_callback):
            while True:
                sd.sleep(int(CHUNK_DURATION * 1000))  # Keep the stream alive
    except KeyboardInterrupt:
        print("🛑 Stopped.")




# from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
# model = load_silero_vad()
# wav = read_audio('sample-audio.wav')
# speech_timestamps = get_speech_timestamps(
#   wav,
#   model,
#   return_seconds=True,  # Return speech timestamps in seconds (default is samples)
# )
#
# print(speech_timestamps)
