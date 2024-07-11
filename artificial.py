import pyttsx3 as p
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import io
import soundfile as sf

# Initialize the pyttsx3 engine
engine = p.init()

# Get and print all available voices
voices = engine.getProperty('voices')
for index, voice in enumerate(voices):
    print(f"Voice {index}: {voice.name} - {voice.id}")

# Set the voice to a known female voice by ID (replace with the actual ID from your system)
voice_id = 'com.apple.speech.synthesis.voice.samantha'  # Replace with your own voice ID
engine.setProperty('voice', voice_id)

# Set the speech rate
rate = engine.getProperty('rate')
engine.setProperty('rate', 180)
print(f"Current speech rate: {rate}")

# Function to speak and print the text
def speak_and_print(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech using the microphone
def recognize_speech():
    recognizer = sr.Recognizer()
    attempts = 0
    while attempts < 3:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
                attempts += 1
            except sr.RequestError:
                print("Could not request results from Google Speech Recognition service.")
                attempts += 1
    return None

# Extended knowledge base
knowledge_base = {
    "name": "Sir Newton",
    "hobby": "watching movies",
    "location": "Spain",
    "age": "not specified",
    "favorite_color": "rainbow (because why settle for one?)",
    "capitals": {
        "france": "Paris",
        "germany": "Berlin",
        "italy": "Rome",
        "spain": "Madrid",
        "portugal": "Lisbon",
        "united kingdom": "London",
        "japan": "Tokyo",
        "china": "Beijing",
        "india": "New Delhi",
        "united states": "Washington, D.C.",
        "kenya": "Nairobi"
    },
    "vehicle_companies": {
        "toyota": "Japan",
        "ford": "United States",
        "bmw": "Germany",
        "audi": "Germany",
        "hyundai": "South Korea",
        "honda": "Japan",
        "chevrolet": "United States",
        "mercedes-benz": "Germany",
        "tesla": "United States",
        "volvo": "Sweden"
    },
    "presidents": {
        "france": "Emmanuel Macron",
        "germany": "Frank-Walter Steinmeier",
        "italy": "Sergio Mattarella",
        "spain": "Pedro Sánchez",
        "portugal": "Marcelo Rebelo de Sousa",
        "united kingdom": "Rishi Sunak (Prime Minister)",
        "japan": "Fumio Kishida (Prime Minister)",
        "china": "Xi Jinping",
        "india": "Droupadi Murmu",
        "united states": "Joe Biden",
        "kenya": "William Ruto"
    },
    "other_knowledge": {
        "largest_ocean": "The Pacific Ocean is the largest ocean on Earth.",
        "tallest_mountain": "Mount Everest is the tallest mountain in the world.",
        "longest_river": "The Nile River is the longest river in the world."
    }
}

# List to store the questions asked
questions_asked = []

# Function to answer questions based on the knowledge base
def answer_question(question):
    question = question.lower()
    tokens = word_tokenize(question)
    
    # Debug: print the tokens
    print(f"Tokens: {tokens}")

    # Store the question
    questions_asked.append(question)

    if any(word in tokens for word in ["name"]):
        return f"My name is {knowledge_base['name']}."
    elif any(word in tokens for word in ["hobby", "like"]):
        return f"I enjoy {knowledge_base['hobby']}."
    elif any(word in tokens for word in ["where", "location"]):
        return f"You can find me {knowledge_base['location']}."
    elif "favorite" in tokens and "color" in tokens:
        return f"My favorite color is {knowledge_base['favorite_color']}."
    elif "age" in tokens or "old" in tokens:
        return f"My age is {knowledge_base['age']}."
    elif "capital" in tokens:
        for country, capital in knowledge_base['capitals'].items():
            if country in tokens:
                return f"The capital of {country.capitalize()} is {capital}."
    elif any(word in tokens for word in ["company", "car", "vehicle"]):
        for company, origin in knowledge_base['vehicle_companies'].items():
            if company in tokens:
                return f"{company.capitalize()} is a vehicle company from {origin}."
    elif any(company in tokens for company in knowledge_base['vehicle_companies']):
        for company, origin in knowledge_base['vehicle_companies'].items():
            if company in tokens:
                return f"{company.capitalize()} originated in {origin}."
    elif "president" in tokens or "leader" in tokens:
        for country, president in knowledge_base['presidents'].items():
            if country in tokens:
                return f"The president of {country.capitalize()} is {president}."
    elif "largest" in tokens and "ocean" in tokens:
        return knowledge_base['other_knowledge']['largest_ocean']
    elif "tallest" in tokens and "mountain" in tokens:
        return knowledge_base['other_knowledge']['tallest_mountain']
    elif "longest" in tokens and "river" in tokens:
        return knowledge_base['other_knowledge']['longest_river']
    elif "questions" in tokens and "asked" in tokens:
        return f"You have asked: {', '.join(questions_asked)}"
    else:
        return "I am not sure about that. Can you ask something else?"

# Download the necessary NLTK data files
nltk.download('punkt')

# Load the VGGish model from TensorFlow Hub
model_url = "https://tfhub.dev/google/vggish/1"
model = hub.load(model_url)

def classify_background_sound():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for background sound...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    # Save audio to a buffer
    audio_buffer = io.BytesIO(audio.get_wav_data())
    waveform, sample_rate = sf.read(audio_buffer)
    
    # Resample the audio to 16kHz
    waveform = waveform.mean(axis=1)  # Convert to mono
    waveform = tf.signal.resample(waveform, 16000)
    
    # Convert to the format expected by the model
    waveform = np.expand_dims(waveform, axis=0)
    embedding = model(waveform)

    # Interpret the embedding
    # Here, you need a method to convert the embedding to a human-readable label.
    # You can use an additional classification model trained on the embeddings to get the label.
    # For simplicity, let's assume we have a `get_sound_label` function.
    sound_label = get_sound_label(embedding.numpy())
    return f"I hear {sound_label} in the background."

def get_sound_label(embedding):
    # This function should convert the embedding to a human-readable label.
    # This is a placeholder implementation and needs to be replaced with actual logic.
    # For example, you could train a classifier on top of the embeddings.
    # Here, we simply return a placeholder string.
    return "a sound"

# Initialize greeting
speak_and_print("Good Morning! I'm Newton. How may I help you?")

# Loop to interact with the user
while True:
    user_input = recognize_speech()
    if user_input is None:
        speak_and_print("Sorry, I did not catch that. Please try again.")
        continue
    if user_input.lower() in ["exit", "quit", "bye"]:
        speak_and_print("Goodbye! Have a fantastic day!")
        break
    elif "thank you" in user_input.lower():
        speak_and_print("You're welcome! Have a lovely day!")
    elif "background sound" in user_input.lower():
        sound_description = classify_background_sound()
        speak_and_print(sound_description)
    else:
        response = answer_question(user_input)
        speak_and_print(response)
