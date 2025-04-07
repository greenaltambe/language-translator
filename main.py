from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.core.text import LabelBase
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3
import whisper
from transformers import pipeline

# Register Hindi-compatible font
LabelBase.register(name="HindiFont", fn_regular="assets/fonts/NotoSansDevanagari-Regular.ttf")

# Load models
whisper_model = whisper.load_model("base")
translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

class TranslatorLayout(BoxLayout):
    def record(self):
        duration = 5  # seconds
        fs = 8000
        self.ids.transcribed_label.text = "Listening..."
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write("audio.wav", fs, audio)
        self.transcribe("audio.wav")

    def transcribe(self, filepath):
        result = whisper_model.transcribe(filepath)
        original_text = result["text"]
        self.ids.transcribed_label.text = original_text
        self.translate(original_text)

    def translate(self, text):
        translated = translator_pipeline(text)[0]["translation_text"]
        self.ids.translated_label.text = translated
        self.speak(translated)

    def speak(self, text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

class TranslatorApp(App):
    def build(self):
        return TranslatorLayout()

if __name__ == '__main__':
    TranslatorApp().run()
