import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sounddevice as sd
import librosa
import joblib
import pyttsx3
import random
from pydub import AudioSegment
from pydub.generators import Sine
from sklearn.preprocessing import StandardScaler
from scipy.io.wavfile import write

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

    plt.show()

# Provide psychologist-like feedback
def provide_psychologist_feedback(predicted_emotion):
    emotion_name = emotions[predicted_emotion]
    emotion_emoji = emojis[emotion_name]
    message = random.choice(psychologist_responses[emotion_name])
    print(f"Emotion: {emotion_name} {emotion_emoji}")
    print("\nPsychologist's Message:")
    print(message)
    engine.say(message)
    engine.runAndWait()

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

# Main function for real-time emotion recognition and psychologist feedback
def main():
    greetings = [
        "Hello there! I'm here to listen and support you.",
        "Hi! How are you feeling today?",
        "Hey! Let's talk about how you're doing."
    ]
    farewells = [
        "Goodbye! Take care and stay positive!",
        "Bye! I'm here whenever you need me.",
        "See you soon! Remember, I'm always here to listen."
    ]
    print(random.choice(greetings))
    engine.say(random.choice(greetings))
    engine.runAndWait()

    while True:
        file_path = record_audio()
        predicted_emotion, prediction_probs = predict_emotion(file_path)

        print(f"Predicted Emotion: {emotions[predicted_emotion]}")
        plot_emotion_probabilities(prediction_probs, emotions, predicted_emotion)

        # Provide psychologist feedback
        provide_psychologist_feedback(predicted_emotion)

        # Generate and play calming music based on the predicted emotion
        generate_calming_music(predicted_emotion)

        # Ask if the user wants to continue
        response = input("Do you want to speak again? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            break

    print(random.choice(farewells))
    engine.say(random.choice(farewells))
    engine.runAndWait()

if __name__ == "__main__":
    main()
