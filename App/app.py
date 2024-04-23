import streamlit as st
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the CountVectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to predict speaker for each sentence
def predict_speaker(sentences):
    predictions = []
    for sentence in sentences:
        # Perform any necessary preprocessing on the sentence before passing it to the model
        # Assuming your model expects a certain format of input features
        
        # Vectorize the sentence using the pre-trained CountVectorizer
        sentence_vec = vectorizer.transform([sentence])
        
        # Predict the speaker for the sentence
        predicted_speaker = model.predict(sentence_vec)[0]
        
        predictions.append((predicted_speaker, sentence))
    return predictions

# Function to transcribe audio to text
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            st.error("Error: Speech recognition could not understand the audio.")
            text = ""
    return text


# Function to separate sentences and predict speakers
def process_audio(audio_file):
    text = transcribe_audio(audio_file)
    sentences = text.split('.')
    predictions = predict_speaker(sentences)
    return predictions

# Function to save conversation to a text file
def save_conversation(predictions, filename='conversation.txt'):
    with open(filename, 'w') as file:
        for prediction in predictions:
            speaker, sentence = prediction
            file.write(f"{speaker}: {sentence.strip()}\n")

# Streamlit UI
st.title("Doctor-Patient Conversation Analyzer")

st.write("Please click 'Start Recording' to begin recording your conversation.")

if st.button("Start Recording"):
    duration = 10  # Adjust duration as needed
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    st.write("Recording...")

    sd.wait()
    audio_filename = "recorded_audio.wav"
    sf.write(audio_filename, recording, fs)
    st.write("Recording stopped.")

    st.write("Processing audio...")
    predictions = process_audio(audio_filename)
    save_conversation(predictions)
    st.write("Conversation saved successfully.")
