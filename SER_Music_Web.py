from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sounddevice as sd
import librosa
import joblib
import pyttsx3
import random
from sklearn.preprocessing import StandardScaler
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
import io
import base64

app = Flask(__name__)

# Initialize the TTS engine
engine = pyttsx3.init()

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


# Plot emotion probabilities
def plot_emotion_probabilities(emotion_probs, emotions, predicted_emotion):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=emotions, y=emotion_probs, palette="viridis")
    plt.xlabel('Emotions', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title('Emotion Probabilities', fontsize=16)
    plt.ylim(0, 1)

    # Display the predicted emotion with an emoji
    emotion_emoji = emojis[emotions[predicted_emotion]]
    plt.text(predicted_emotion, emotion_probs[predicted_emotion] + 0.05, emotion_emoji, fontsize=20, ha='center')

    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64


# Provide motivational feedback and generate calming music
def provide_motivational_feedback(predicted_emotion):
    emotion_name = emotions[predicted_emotion]
    emotion_emoji = emojis[emotion_name]
    message = random.choice(motivational_messages[emotion_name])
    print(f"Emotion: {emotion_name} {emotion_emoji}")
    print("\nMotivational Message:")
    print(message)
    engine.say(message)
    engine.runAndWait()

    return message


# Generate calming music based on the detected emotion
def generate_calming_music(emotion):
    base_freq = {
        'angry': 220,  # A3
        'disgust': 233,  # Bb3
        'fear': 261,  # C4
        'happy': 293,  # D4
        'neutral': 329,  # E4
        'sad': 349,  # F4
        'surprise': 392  # G4
    }
    duration = 5000  # 5 seconds
    frequency = base_freq.get(emotion, 440)  # Default to 440 Hz (A4)

    # Generate a sine wave
    sine_wave = Sine(frequency).to_audio_segment(duration=duration)
    sine_wave = sine_wave.fade_in(1000).fade_out(1000)  # Fade in/out for smoothness

    # Play the generated sine wave
    play(sine_wave)


@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    file_path = record_audio()
    predicted_emotion, prediction_probs = predict_emotion(file_path)

    print(f"Predicted Emotion: {emotions[predicted_emotion]}")
    img_base64 = plot_emotion_probabilities(prediction_probs, emotions, predicted_emotion)

    # Provide motivational feedback
    message = provide_motivational_feedback(predicted_emotion)

    return jsonify({
        'emotion': emotions[predicted_emotion],
        'message': message,
        'image': img_base64
    })
@app.route('/record', methods=['POST'])
def record():
    file_path = record_audio()
    predicted_emotion, _ = predict_emotion(file_path)
    emotion_name = emotions[predicted_emotion]
    message = random.choice(motivational_messages[emotion_name])
    response = {
        'emotion': emotion_name,
        'message': message
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
