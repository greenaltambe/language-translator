from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.core.text import LabelBase
from kivy.clock import Clock

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import pyttsx3
import whisper
from transformers import pipeline

# Register Hindi-compatible font
LabelBase.register(
    name="HindiFont", fn_regular="assets/fonts/NotoSansDevanagari-Regular.ttf"
)

# Load models
whisper_model = whisper.load_model("base")
translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")


class TranslatorLayout(BoxLayout):
    def record(self):
        duration = 5  # seconds
        fs = 16000  # Whisper performs best with 16000 Hz
        self.ids.transcribed_label.color = (1, 0, 0, 1)  # Red while recording
        self.ids.transcribed_label.text = "Recording..."

        def _start_recording(dt):
            audio = sd.rec(
                int(duration * fs), samplerate=fs, channels=1, dtype="float32"
            )
            sd.wait()

            # Normalize audio
            audio = audio / np.max(np.abs(audio))

            write("audio.wav", fs, audio)
            self.ids.transcribed_label.color = (1, 1, 1, 1)  # Reset to white
            self.transcribe("audio.wav")

        # Delay actual recording to allow UI update
        Clock.schedule_once(_start_recording, 0.1)

    def transcribe(self, filepath):
        self.ids.transcribed_label.text = "Transcribing..."
        result = whisper_model.transcribe(filepath)
        original_text = result["text"].strip()
        self.ids.transcribed_label.text = original_text or "Couldn't understand audio."
        if original_text:
            self.translate(original_text)

    def translate(self, text):
        self.ids.translated_label.text = "Translating..."
        translated = translator_pipeline(text)[0]["translation_text"]
        self.ids.translated_label.text = translated
        self.speak(translated)

    def speak(self, text):
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()


class TranslatorApp(App):
    def build(self):
        return TranslatorLayout()


if __name__ == "__main__":
    TranslatorApp().run()
