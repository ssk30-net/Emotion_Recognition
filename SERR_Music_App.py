import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.audio import SoundLoader
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Rectangle  # Import Color and Rectangle from kivy.graphics
import numpy as np
import sounddevice as sd
import librosa
import joblib
import random
from sklearn.preprocessing import StandardScaler
from scipy.io.wavfile import write
from kivy.clock import Clock
import pyttsx3
import os

kivy.require('2.0.0')

# Load the pre-trained Random Forest model
model_path = 'random_forest_model.joblib'
rf_model = joblib.load(model_path)

# Define emotions, their corresponding emojis, and motivational messages
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emojis = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}
motivational_messages = {
    'angry': [
        "Take a deep breath and try to relax. Remember, it's important to stay calm and composed.",
        "When you feel anger rising, try to count to ten before reacting.",
        "Anger is a natural emotion. Use it as a signal to address what's bothering you.",
        "Let's try some deep breathing exercises to calm down."
    ],
    'disgust': [
        "It's okay to feel disgust. Try to focus on something that brings you joy.",
        "Disgust can be a sign of something that doesn't align with your values. Reflect on what that might be.",
        "Take a moment to step away and clear your mind.",
        "Remember to stay positive and focus on the good things in life."
    ],
    'fear': [
        "It's normal to feel fear. Try to take slow, deep breaths to calm your mind.",
        "Fear is often just False Evidence Appearing Real. Challenge your fearful thoughts.",
        "You are stronger than you think. Take small steps to face your fears.",
        "Let's visualize a safe and calm place together."
    ],
    'happy': [
        "Keep spreading your positive energy. It's infectious!",
        "Happiness is a wonderful feeling. Savor and enjoy it.",
        "Your smile is a gift to the world. Keep shining!",
        "Let's celebrate your happiness with some joyful music!"
    ],
    'neutral': [
        "It's okay to feel neutral. Embrace the calmness.",
        "Neutral moments are opportunities for reflection and balance.",
        "Take this time to center yourself and recharge.",
        "Let's find a relaxing activity to enjoy together."
    ],
    'sad': [
        "It's okay to feel sad. Allow yourself to experience your emotions.",
        "Sadness can be overwhelming. Remember to be kind to yourself.",
        "Reach out to someone you trust. Talking about it can help.",
        "Let's try a calming activity to soothe your mind."
    ],
    'surprise': [
        "Embrace the unexpected. Surprises can bring new opportunities.",
        "Life is full of surprises. Stay open to new experiences.",
        "A surprise can be a blessing in disguise. Keep a positive mindset.",
        "Let's enjoy this surprising moment together."
    ]
}

# Function to extract features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Record audio
def record_audio(duration=3, fs=44100):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # Wait until the recording is finished
    file_path = 'recorded_audio.wav'
    write(file_path, fs, recording)
    return file_path

# Predict emotion from audio
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    prediction_probs = rf_model.predict_proba(features_scaled)[0]
    predicted_emotion = rf_model.predict(features_scaled)[0]
    return predicted_emotion, prediction_probs

class EmotionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        self.avatar = Image(source='templates/static/avatar.jpg', size_hint=(None, None), size=(150, 150))
        self.layout.add_widget(self.avatar)

        self.greeting_label = Label(text="Hey, I am Adithya, your AI friend!", font_size='24sp', color=(0, 0, 0, 1))
        self.layout.add_widget(self.greeting_label)

        self.record_button = Button(text="Start Recording", size_hint=(None, None), size=(200, 50), background_color=(0, 0.5, 1, 1))
        self.record_button.bind(on_press=self.on_record)
        self.layout.add_widget(self.record_button)

        self.emotion_label = Label(text="", font_size='20sp', color=(0, 0, 0, 1))
        self.layout.add_widget(self.emotion_label)

        self.motivational_label = Label(text="", font_size='18sp', halign='center', color=(0, 0, 0, 1))
        self.layout.add_widget(self.motivational_label)

        self.graph_layout = GridLayout(cols=len(emotions), spacing=10, size_hint=(1, 0.5))
        self.layout.add_widget(self.graph_layout)

        return self.layout

    def on_record(self, instance):
        self.record_button.text = "Recording..."
        self.record_button.background_color = (1, 0, 0, 1)
        Clock.schedule_once(self.stop_recording, 3)

    def stop_recording(self, dt):
        file_path = record_audio()
        predicted_emotion, prediction_probs = predict_emotion(file_path)

        emotion_name = emotions[predicted_emotion]
        message = random.choice(motivational_messages[emotion_name])

        self.emotion_label.text = f"Detected Emotion: {emotion_name.capitalize()} {emojis[emotion_name]}"
        self.motivational_label.text = f"{message}"

        self.record_button.text = "Start Recording"
        self.record_button.background_color = (0, 0.5, 1, 1)

        self.speak(message)
        self.update_graph(prediction_probs)
        self.play_calming_music(emotion_name)
        self.change_background(emotion_name)

    def speak(self, text):
        engine = pyttsx3.init()
        engine.save_to_file(text, 'speech.mp3')
        engine.runAndWait()
        sound = SoundLoader.load('speech.mp3')
        if sound:
            sound.play()
            Clock.schedule_once(lambda dt: self.delete_file('speech.mp3'), sound.length)

    def delete_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def update_graph(self, prediction_probs):
        self.graph_layout.clear_widgets()

        for i, emotion in enumerate(emotions):
            emotion_info = BoxLayout(orientation='vertical', spacing=5)

            emotion_img = Image(source=f'{emotion}.png', size_hint=(None, None), size=(100, 100))
            emotion_info.add_widget(emotion_img)

            emotion_label = Label(text=f'{emotion.capitalize()} {emojis[emotion]}', font_size='16sp',
                                  color=(0, 0, 0, 1))
            emotion_info.add_widget(emotion_label)

            percentage_label = Label(text=f'{prediction_probs[i] * 100:.2f}%', font_size='14sp', color=(0, 0, 0, 1))
            emotion_info.add_widget(percentage_label)

            self.graph_layout.add_widget(emotion_info)

    def play_calming_music(self, emotion):
        frequencies = {
            'angry': 220,
            'disgust': 233,
            'fear': 261,
            'happy': 293,
            'neutral': 329,
            'sad': 349,
            'surprise': 392
        }
        duration = 5
        frequency = frequencies.get(emotion, 440)
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)

        temp_file = 'calming_music.wav'
        write(temp_file, sample_rate, audio.astype(np.float32))

        sound = SoundLoader.load(temp_file)
        if sound:
            sound.play()
            Clock.schedule_once(lambda dt: self.delete_file(temp_file), sound.length)

    def change_background(self, emotion):
        colors = {
            'angry': (1, 0.3, 0.3, 1),
            'disgust': (0.6, 0.6, 0.6, 1),
            'fear': (0.5, 0.5, 1, 1),
            'happy': (1, 1, 0.5, 1),
            'neutral': (0.8, 0.8, 0.8, 1),
            'sad': (0.5, 0.5, 0.8, 1),
            'surprise': (1, 0.8, 0.5, 1)
        }
        self.layout.canvas.before.clear()
        with self.layout.canvas.before:
            Color(*colors.get(emotion, (1, 1, 1, 1)))
            Rectangle(pos=self.layout.pos, size=self.layout.size)

if __name__ == '__main__':
    EmotionApp().run()

