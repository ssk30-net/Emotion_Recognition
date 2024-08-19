import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import defaultdict


# Function to generate psychologist-like responses
def generate_responses(emotion):
    responses = {
        "angry": [
            "It seems like you're feeling quite angry. Would you like to talk about what's causing this anger?",
            "Anger can be a powerful emotion. What's been happening that's made you feel this way?",
            "Sometimes anger can be overwhelming. What's on your mind?"
        ],
        "disgust": [
            "Feeling disgusted can be very uncomfortable. What do you think is triggering this feeling?",
            "Disgust is a strong emotion. Can you tell me more about what's making you feel this way?",
            "It sounds like something has really upset you. Do you want to talk about it?"
        ],
        "fear": [
            "It sounds like you're feeling afraid. Is there something specific that is worrying you?",
            "Fear can be a challenging emotion to deal with. What do you think is causing this fear?",
            "It's okay to feel scared. What's been happening that's making you feel this way?"
        ],
        "happy": [
            "I'm glad to hear that you're feeling happy! What is making you feel this way?",
            "It's great to see you're in a good mood! Want to share more about your day?",
            "Happiness can be very uplifting. What's bringing joy to your life right now?"
        ],
        "neutral": [
            "It seems like you're feeling neutral. Is there anything on your mind you'd like to discuss?",
            "You seem to be in a balanced mood. Would you like to talk about anything specific?",
            "Feeling neutral can sometimes be a good thing. How has your day been?"
        ],
        "sad": [
            "I'm sorry to hear that you're feeling sad. Do you want to share what's on your mind?",
            "It sounds like you're feeling down. What's been happening?",
            "Sadness can be tough to deal with. I'm here to listen if you want to talk."
        ],
        "surprise": [
            "It sounds like something surprising happened. Do you want to tell me more about it?",
            "Surprise can be both good and bad. What's been going on?",
            "You seem surprised. Want to share what happened?"
        ]
    }
    return responses.get(emotion, ["How are you feeling?"])


# Function to analyze emotions and generate response
def analyze_emotion_and_respond(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Keywords for each emotion
    emotion_keywords = {
        "angry": ["angry", "mad", "furious", "rage"],
        "disgust": ["disgust", "revolting", "nauseating"],
        "fear": ["fear", "afraid", "scared", "terrified"],
        "happy": ["happy", "joy", "delighted", "pleased"],
        "neutral": ["okay", "fine", "neutral"],
        "sad": ["sad", "down", "unhappy", "depressed"],
        "surprise": ["surprise", "shocked", "amazed", "astonished"]
    }

    # Detect emotion based on keywords
    detected_emotions = defaultdict(int)
    for word in text.lower().split():
        for emotion, keywords in emotion_keywords.items():
            if word in keywords:
                detected_emotions[emotion] += 1

    # Determine primary emotion
    primary_emotion = max(detected_emotions, key=detected_emotions.get) if detected_emotions else "neutral"

    responses = generate_responses(primary_emotion)

    return primary_emotion, detected_emotions, responses


# Function to plot emotions
def plot_emotions(emotions):
    plt.figure(figsize=(10, 5))
    plt.bar(emotions.keys(), emotions.values(), color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Frequency')
    plt.title('Detected Emotions')
    plt.show()


# Example usage
if __name__ == "__main__":
    user_input = input("How are you feeling today? ")
    primary_emotion, detected_emotions, responses = analyze_emotion_and_respond(user_input)

    print(f"Detected Emotion: {primary_emotion}")
    print("Responses from Psychologist:")
    for response in responses:
        print(f"- {response}")

    plot_emotions(detected_emotions)
