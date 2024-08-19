import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
import sounddevice as sd
import librosa
import joblib
import random
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler
import os
import pyttsx3
import numpy as np

kivy.require('2.0.0')

# Load the pre-trained Random Forest model
model_path = 'random_forest_model.joblib'
rf_model = joblib.load(model_path)

# Define emotions and motivational messages
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
        "Take a deep breath and try to relax. Remember, challenges are opportunities in disguise.",
        "Anger is an acid that can do more harm to the vessel in which it is stored than to anything on which it is poured.",
        "It's okay to feel angry. Take a moment to breathe deeply and calm your mind.",
        "Hey, friend! Let's cool down together. How about taking a walk?"
    ],
    'disgust': [
        "Focus on the positive aspects and find things that make you happy.",
        "Sometimes we encounter things we don't like, but it's important to stay positive.",
        "Disgust is a natural feeling. Try to shift your focus to something pleasant.",
        "Hey buddy, let's find something awesome to cheer you up!"
    ],
    'fear': [
        "You are stronger than you think. Face your fears with courage and confidence.",
        "Fear is just a state of mind. You have the power to overcome it.",
        "Embrace your fears and turn them into your strengths.",
        "I'm here with you, let's face this fear together!"
    ],
    'happy': [
        "Keep spreading the joy! Your happiness is contagious.",
        "Happiness is a choice. Keep choosing to be happy.",
        "Your positive energy is inspiring. Keep shining!",
        "Yay! Your happiness makes my day too!"
    ],
    'neutral': [
        "Stay balanced and keep up the good work. You're doing great!",
        "Neutral moments are a part of life. Stay positive and focused.",
        "Keep maintaining your balance and inner peace.",
        "Hey, let's find something fun to make your day brighter!"
    ],
    'sad': [
        "It's okay to feel sad sometimes. Take care of yourself and reach out to loved ones.",
        "Sadness is a part of life. Remember, after the rain comes the rainbow.",
        "Allow yourself to feel the sadness, but also remember to embrace the joy in life.",
        "I'm here for you, friend. Let's talk about what's on your mind."
    ],
    'surprise': [
        "Embrace the unexpected and enjoy the surprises that life brings.",
        "Life is full of surprises. Keep an open mind and enjoy the journey.",
        "Surprises can be wonderful opportunities. Stay positive and hopeful.",
        "Wow! What a surprise! Let's enjoy this moment together."
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
        self.title = 'Emotion Recognition'
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.avatar = Image(source='templates/static/avatar.jpg', size_hint=(1, 0.3))
        self.layout.add_widget(self.avatar)

        self.button = Button(text='Start Recording', size_hint=(1, 0.1))
        self.button.bind(on_press=self.record_emotion)
        self.layout.add_widget(self.button)

        self.emotion_label = Label(text='', size_hint=(1, 0.1), font_size='20sp')
        self.layout.add_widget(self.emotion_label)

        self.message_label = Label(text='', size_hint=(1, 0.2), font_size='16sp')
        self.layout.add_widget(self.message_label)

        self.graph_layout = GridLayout(cols=len(emotions), spacing=10, size_hint=(1, 0.4))
        self.layout.add_widget(self.graph_layout)

        return self.layout

    def record_emotion(self, instance):
        file_path = record_audio()
        predicted_emotion, prediction_probs = predict_emotion(file_path)
        emotion_name = emotions[predicted_emotion]
        message = random.choice(motivational_messages[emotion_name])

        self.emotion_label.text = f'Emotion: {emotion_name.capitalize()} {emojis[emotion_name]}'
        self.message_label.text = f'Message: {message}'

        self.speak(message)
        self.update_graph(prediction_probs)

    def speak(self, text):
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Save the speech to a file
        engine.save_to_file(text, 'speech.mp3')
        engine.runAndWait()

        # Load and play the speech
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

            emotion_label = Label(text=f'{emotion.capitalize()} {emojis[emotion]}', font_size='16sp')
            emotion_info.add_widget(emotion_label)

            percentage_label = Label(text=f'{prediction_probs[i] * 100:.2f}%', font_size='14sp')
            emotion_info.add_widget(percentage_label)

            self.graph_layout.add_widget(emotion_info)


if __name__ == '__main__':
    EmotionApp().run()
