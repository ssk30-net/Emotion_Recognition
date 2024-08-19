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

    plt.show()


# Provide motivational feedback
def provide_motivational_feedback(predicted_emotion):
    emotion_name = emotions[predicted_emotion]
    emotion_emoji = emojis[emotion_name]
    message = random.choice(motivational_messages[emotion_name])
    print(f"Emotion: {emotion_name} {emotion_emoji}")
    print("\nMotivational Message:")
    print(message)
    engine.say(message)
    engine.runAndWait()


# Main function for real-time emotion recognition and motivational feedback
def main():
    greetings = [
        "Hello there! I'm here to cheer you up!",
        "Hi! How are you feeling today?",
        "Hey! Let's have a great conversation!"
    ]
    farewells = [
        "Goodbye! Take care and stay positive!",
        "Bye! I'm here whenever you need me!",
        "See you soon! Keep smiling!"
    ]
    print(random.choice(greetings))
    engine.say(random.choice(greetings))
    engine.runAndWait()

    while True:
        file_path = record_audio()
        predicted_emotion, prediction_probs = predict_emotion(file_path)

        print(f"Predicted Emotion: {emotions[predicted_emotion]}")
        plot_emotion_probabilities(prediction_probs, emotions, predicted_emotion)

        # Provide motivational feedback
        provide_motivational_feedback(predicted_emotion)

        # Ask if the user wants to continue
        response = input("Do you want to speak again? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            break

    print(random.choice(farewells))
    engine.say(random.choice(farewells))
    engine.runAndWait()


if __name__ == "__main__":
    main()
