import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sounddevice as sd
import librosa
import joblib
import pyttsx3
from sklearn.preprocessing import StandardScaler
from scipy.io.wavfile import write
from io import BytesIO
from pydub import AudioSegment
from pydub.generators import Sine
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QDialog, QLabel, QHBoxLayout, QFileDialog, QGridLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import matplotlib
from textblob import TextBlob
from collections import defaultdict
matplotlib.use('Qt5Agg')

# Initialize the TTS engine
engine = pyttsx3.init()

# Load the pre-trained Random Forest model
model_path = 'random_forest_model.joblib'
rf_model = joblib.load(model_path)

# Define emotions, their corresponding emojis, and psychologist-like responses
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
psychologist_responses = {
    'angry': [
        "It seems like you're feeling quite angry. Would you like to talk about what's causing this anger?",
        "Anger can be a powerful emotion. What's been happening that's made you feel this way?",
        "Sometimes anger can be overwhelming. What's on your mind?",
        "I hear that you're angry. Can you share more about what's triggering this feeling?"
    ],
    'disgust': [
        "Feeling disgusted can be very uncomfortable. What do you think is triggering this feeling?",
        "Disgust is a strong emotion. Can you tell me more about what's making you feel this way?",
        "It sounds like something has really upset you. Do you want to talk about it?",
        "What is it that's causing you to feel this way?"
    ],
    'fear': [
        "It sounds like you're feeling afraid. Is there something specific that is worrying you?",
        "Fear can be a challenging emotion to deal with. What do you think is causing this fear?",
        "It's okay to feel scared. What's been happening that's making you feel this way?",
        "Can you tell me more about what you're feeling afraid of?"
    ],
    'happy': [
        "I'm glad to hear that you're feeling happy! What is making you feel this way?",
        "It's great to see you're in a good mood! Want to share more about your day?",
        "Happiness can be very uplifting. What's bringing joy to your life right now?",
        "What is it that's making you feel so happy?"
    ],
    'neutral': [
        "It seems like you're feeling neutral. Is there anything on your mind you'd like to discuss?",
        "You seem to be in a balanced mood. Would you like to talk about anything specific?",
        "Feeling neutral can sometimes be a good thing. How has your day been?",
        "What's going on in your mind right now?"
    ],
    'sad': [
        "I'm sorry to hear that you're feeling sad. Do you want to share what's on your mind?",
        "It sounds like you're feeling down. What's been happening?",
        "Sadness can be tough to deal with. I'm here to listen if you want to talk.",
        "Can you tell me more about why you're feeling sad?"
    ],
    'surprise': [
        "It sounds like something surprising happened. Do you want to tell me more about it?",
        "Surprise can be both good and bad. What's been going on?",
        "You seem surprised. Want to share what happened?",
        "What has surprised you recently?"
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
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
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

# Plot emotion probabilities
def plot_emotion_probabilities(emotion_probs, emotions, predicted_emotion):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=emotions, y=emotion_probs)
    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.title('Emotion Probabilities')
    plt.ylim(0, 1)

    # Display the predicted emotion with an emoji
    emotion_emoji = emojis[emotions[predicted_emotion]]
    plt.text(predicted_emotion, emotion_probs[predicted_emotion] + 0.05, emotion_emoji, fontsize=20, ha='center')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Plot pie chart of emotion probabilities
def plot_emotion_pie_chart(emotion_probs, emotions, predicted_emotion):
    plt.figure(figsize=(8, 8))
    colors = sns.color_palette('pastel')
    explode = [0] * len(emotions)
    explode[predicted_emotion] = 0.1  # Explode the predicted emotion slice

    plt.pie(emotion_probs, labels=emotions, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Emotion Probabilities')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Provide psychologist-like feedback
def provide_psychologist_feedback(predicted_emotion):
    emotion_name = emotions[predicted_emotion]
    emotion_emoji = emojis[emotion_name]
    message = random.choice(psychologist_responses[emotion_name])
    return f"Emotion: {emotion_name} {emotion_emoji}\n\nPsychologist's Message:\n{message}"

# Generate calming music based on the predicted emotion
def generate_calming_music(predicted_emotion):
    emotion_name = emotions[predicted_emotion]

    # Define the frequencies and duration for each emotion
    freq_map = {
        'angry': [220, 440],  # Lower frequencies
        'disgust': [330, 660],  # Medium frequencies
        'fear': [220, 330],  # Lower frequencies
        'happy': [440, 880],  # Higher frequencies
        'neutral': [330, 440],  # Medium frequencies
        'sad': [220, 330],  # Lower frequencies
        'surprise': [440, 660]  # Medium to higher frequencies
    }
    duration = 1000  # Duration of each tone in milliseconds

    music_segment = AudioSegment.silent(duration=0)

    for freq in freq_map[emotion_name]:
        tone = Sine(freq).to_audio_segment(duration=duration)
        music_segment += tone

    music_file = f"generated_music_{emotion_name}.wav"
    music_segment.export(music_file, format="wav")
    print(f"Generated calming music for {emotion_name} and saved as {music_file}")

    # Play the generated music
    sd.play(music_segment.get_array_of_samples(), samplerate=44100)
    sd.wait()

    # Function to plot emotion probabilities in a pie chart
    def plot_emotion_pie_chart_text(predicted_emotion):
        # Create a figure and set it up
        plt.figure(figsize=(8, 8))
        labels = emotions
        sizes = [0] * len(emotions)
        sizes[predicted_emotion] = 1  # Set the predicted emotion size to 1, others to 0

        # Define colors and explode the slice of the pie chart for emphasis
        colors = sns.color_palette('pastel')
        explode = [0] * len(emotions)
        explode[predicted_emotion] = 0.1  # Explode the predicted emotion slice

        # Plot the pie chart
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=explode)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the plot to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Recognition App")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #f0f0f0;")

        layout = QGridLayout()

        self.text_button = QPushButton("Text-Based Emotion Detection")
        self.text_button.setFont(QFont('Arial', 14))
        self.text_button.setStyleSheet("padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.text_button.clicked.connect(self.show_text_input)
        layout.addWidget(self.text_button, 0, 0, 1, 1)

        self.speech_button = QPushButton("Speech-Based Emotion Detection")
        self.speech_button.setFont(QFont('Arial', 14))
        self.speech_button.setStyleSheet("padding: 10px; background-color: #008CBA; color: white; border-radius: 5px;")
        self.speech_button.clicked.connect(self.show_speech_input)
        layout.addWidget(self.speech_button, 0, 1, 1, 1)

        self.setLayout(layout)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def show_text_input(self):
        dialog = TextInputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            text = dialog.get_text()
            self.analyze_text_emotion(text)

    def show_speech_input(self):
        dialog = SpeechInputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.record_and_analyze_speech()

    def analyze_text_emotion(self, text):
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity

        # Map sentiment polarity to emotions
        if sentiment_polarity > 0.2:
            predicted_emotion = emotions.index('happy')
        elif sentiment_polarity > -0.2 and sentiment_polarity <= 0.2:
            predicted_emotion = emotions.index('neutral')
        else:
            predicted_emotion = emotions.index('sad')

        # Generate psychologist-like feedback
        results_text = provide_psychologist_feedback(predicted_emotion)

        # Dummy prediction probabilities (replace with actual logic if needed)
        prediction_probs = np.random.rand(len(emotions))
        prediction_probs /= prediction_probs.sum()

        buf = plot_emotion_pie_chart(prediction_probs, emotions, predicted_emotion)

        dialog = EmotionResultsDialog(self, results_text, buf)
        dialog.exec_()

    def record_and_analyze_speech(self):
        file_path = record_audio()
        predicted_emotion, prediction_probs = predict_emotion(file_path)
        results_text = provide_psychologist_feedback(predicted_emotion)
        buf = plot_emotion_pie_chart(prediction_probs, emotions, predicted_emotion)

        dialog = EmotionResultsDialog(self, results_text, buf)
        dialog.exec_()

class TextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text-Based Emotion Detection")
        self.setGeometry(200, 200, 400, 300)
        self.setStyleSheet("background-color: #f0f0f0;")

        layout = QVBoxLayout()

        self.text_edit = QTextEdit(self)
        layout.addWidget(self.text_edit)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.setFont(QFont('Arial', 12))
        self.ok_button.setStyleSheet("padding: 5px; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFont(QFont('Arial', 12))
        self.cancel_button.setStyleSheet("padding: 5px; background-color: #f44336; color: white; border-radius: 5px;")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_text(self):
        return self.text_edit.toPlainText()

class SpeechInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Speech-Based Emotion Detection")
        self.setGeometry(200, 200, 400, 300)
        self.setStyleSheet("background-color: #f0f0f0;")

        layout = QVBoxLayout()

        self.label = QLabel("Press 'Record' to start recording your speech.", self)
        self.label.setFont(QFont('Arial', 12))
        layout.addWidget(self.label)

        self.record_button = QPushButton("Record")
        self.record_button.setFont(QFont('Arial', 12))
        self.record_button.setStyleSheet("padding: 5px; background-color: #008CBA; color: white; border-radius: 5px;")
        self.record_button.clicked.connect(self.accept)
        layout.addWidget(self.record_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFont(QFont('Arial', 12))
        self.cancel_button.setStyleSheet("padding: 5px; background-color: #f44336; color: white; border-radius: 5px;")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

class EmotionResultsDialog(QDialog):
    def __init__(self, parent, results_text, plot_buf):
        super().__init__(parent)
        self.setWindowTitle("Emotion Analysis Results")
        self.setGeometry(200, 200, 600, 600)
        self.setStyleSheet("background-color: #f0f0f0;")

        layout = QVBoxLayout()

        self.results_label = QLabel(results_text, self)
        self.results_label.setWordWrap(True)
        self.results_label.setFont(QFont('Arial', 14))
        layout.addWidget(self.results_label)

        self.plot_label = QLabel(self)
        pixmap = QPixmap()
        pixmap.loadFromData(plot_buf.getvalue())
        self.plot_label.setPixmap(pixmap)
        layout.addWidget(self.plot_label)

        self.music_button = QPushButton("Generate Calming Music")
        self.music_button.setFont(QFont('Arial', 14))
        self.music_button.setStyleSheet("padding: 10px; background-color: #FF9800; color: white; border-radius: 5px;")
        self.music_button.clicked.connect(self.generate_music)
        layout.addWidget(self.music_button)

        self.close_button = QPushButton("Close")
        self.close_button.setFont(QFont('Arial', 14))
        self.close_button.setStyleSheet("padding: 10px; background-color: #f44336; color: white; border-radius: 5px;")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

    def generate_music(self):
        predicted_emotion = random.choice(range(len(emotions)))  # Replace with actual predicted emotion
        generate_calming_music(predicted_emotion)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
