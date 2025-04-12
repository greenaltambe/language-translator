from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.core.text import LabelBase
from kivy.clock import Clock

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3
import whisper

import argostranslate.package
import argostranslate.translate

# Language codes for translation
from_code = "en"
to_code = "hi"

LabelBase.register(
    name="HindiFont",
    fn_regular="assets/fonts/NotoSansDevanagari-Regular.ttf"
)

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Set up Argos Translate (offline translator)
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code
)
argostranslate.package.install_from_path(package_to_install.download())


class TranslatorLayout(BoxLayout):
    def record(self):
        duration = 5  # seconds
        fs = 16000  # Recommended for Whisper

        self.ids.transcribed_label.color = (1, 0, 0, 1)  # Red during recording
        self.ids.transcribed_label.text = "Recording..."

        def _start_recording(dt):
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
            sd.wait()

            # Normalize audio to prevent clipping
            audio = audio / np.max(np.abs(audio))
            write("audio.wav", fs, audio)

            self.ids.transcribed_label.color = (1, 1, 1, 1)  # Back to white
            self.transcribe("audio.wav")

        # Slight delay to update UI before recording starts
        Clock.schedule_once(_start_recording, 0.1)

    def transcribe(self, filepath):
        self.ids.transcribed_label.text = "Transcribing..."
        result = whisper_model.transcribe(filepath)
        text = result["text"].strip()

        self.ids.transcribed_label.text = text or "Couldn't understand audio."
        if text:
            self.translate(text)

    def translate(self, text):
        self.ids.translated_label.text = "Translating..."
        translated = argostranslate.translate.translate(text, from_code, to_code)
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
