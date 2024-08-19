from flask import Flask, render_template, request, send_from_directory
import numpy as np
import joblib
import librosa
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pyttsx3
from io import BytesIO
import base64
from scipy.io.wavfile import write
import sounddevice as sd
import os

app = Flask(__name__)

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

# Ensure the directory exists for static audio files
audio_dir = 'static/audio'
os.makedirs(audio_dir, exist_ok=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def record_audio(duration=3, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    file_path = os.path.join(audio_dir, 'recorded_audio.wav')
    write(file_path, fs, recording)
    return file_path

def predict_emotion(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    prediction_probs = rf_model.predict_proba(features_scaled)[0]
    predicted_emotion = rf_model.predict(features_scaled)[0]
    return predicted_emotion, prediction_probs

def provide_psychologist_feedback(predicted_emotion):
    emotion_name = emotions[predicted_emotion]
    emotion_emoji = emojis[emotion_name]
    message = random.choice(psychologist_responses[emotion_name])
    return f"Emotion: {emotion_name} {emotion_emoji}\n\nPsychologist's Message:\n{message}"

def plot_emotion_pie_chart(predicted_emotion):
    plt.figure(figsize=(8, 8))
    labels = emotions
    sizes = [0] * len(emotions)
    sizes[predicted_emotion] = 1
    colors = sns.color_palette('pastel')
    explode = [0] * len(emotions)
    explode[predicted_emotion] = 0.1
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=explode)
    plt.axis('equal')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/text_input', methods=['GET', 'POST'])
def text_input():
    if request.method == 'POST':
        text = request.form['text']
        text_lower = text.lower()

        keyword_emotion_mapping = {
            'angry': ['angry', 'annoyed', 'frustrated', 'irate', 'enraged'],
            'disgust': ['disgust', 'revolted', 'repulsed', 'abhor'],
            'fear': ['fear', 'scared', 'afraid', 'terrified'],
            'happy': ['happy', 'joyful', 'cheerful', 'delighted'],
            'neutral': ['neutral', 'indifferent', 'calm', 'unperturbed'],
            'sad': ['sad', 'unhappy', 'gloomy', 'miserable'],
            'surprise': ['surprise', 'shock', 'astonished', 'amazed']
        }

        predicted_emotion = 'neutral'

        for emotion, keywords in keyword_emotion_mapping.items():
            for keyword in keywords:
                if keyword in text_lower:
                    predicted_emotion = emotion
                    break
            if predicted_emotion != 'neutral':
                break

        results_text = provide_psychologist_feedback(emotions.index(predicted_emotion))
        buf = plot_emotion_pie_chart(emotions.index(predicted_emotion))
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        return render_template('text_input.html', results_text=results_text, img_data=img_data, play_music=True)
    return render_template('text_input.html')

@app.route('/speech_input', methods=['GET', 'POST'])
def speech_input():
    if request.method == 'POST':
        file_path = record_audio()
        predicted_emotion, prediction_probs = predict_emotion(file_path)
        results_text = provide_psychologist_feedback(predicted_emotion)
        buf = plot_emotion_pie_chart(predicted_emotion)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        return render_template('speech_input.html', results_text=results_text, img_data=img_data, play_music=True)
    return render_template('speech_input.html')

@app.route('/static/audio/<filename>')
def send_audio(filename):
    return send_from_directory('static/audio', filename)

if __name__ == "__main__":
    app.run(debug=True)
