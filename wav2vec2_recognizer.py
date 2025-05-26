
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import os

class Wav2Vec2SpeechRecognizer:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        print("[INFO] Loading Wav2Vec2 model and processor...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.sampling_rate = 16000

    def record_audio(self, duration=5, filename="temp_audio.wav"):
        print("[INFO] Recording audio...")
        audio = sd.rec(int(duration * self.sampling_rate), samplerate=self.sampling_rate, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        wavfile.write(filename, self.sampling_rate, audio)
        print(f"[INFO] Audio recorded and saved to {filename}")
        return filename

    def transcribe_audio(self, audio_path):
        print(f"[INFO] Transcribing audio from {audio_path}...")
        import librosa
        speech, _ = librosa.load(audio_path, sr=self.sampling_rate)
        input_values = self.processor(speech, return_tensors="pt", sampling_rate=self.sampling_rate).input_values
        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        print("[INFO] Transcription complete.")
        return transcription

if __name__ == "__main__":
    recognizer = Wav2Vec2SpeechRecognizer()
    audio_path = recognizer.record_audio(duration=5)
    text = recognizer.transcribe_audio(audio_path)
    print("Transcribed Text:", text)
