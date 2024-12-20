import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import wavio
import time
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

def extract_feature(file, mel=True):
    # Since we assume a standard recording, we set sr=44100
    y, sr = librosa.load(file, sr=44100)
    if mel:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128).T, axis=0)
    else:
        mfccs = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return mfccs

def classify_gender(file):
    audio_length = librosa.get_duration(filename=file)
    start = time.time()
    features = extract_feature(file, mel=True).reshape(1, -1)
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "Male" if male_prob > female_prob else "Female"
    end = time.time()
    final_time = end - start

    result = {
        "gender": gender,
        "male_probability": f"{male_prob * 100:.2f}%",
        "female_probability": f"{female_prob * 100:.2f}%",
        "time_taken": f"{final_time:.2f} seconds",
        "audio_length": f"{audio_length:.2f} seconds",
    }

    return result

# Streamlit interface
st.title("🎙️ Voice Gender Classifier")
st.write("This model predicts the gender of a speaker from an uploaded audio file.")

upload_mode = st.sidebar.radio("Audio Input Method", ("File Upload",))

if upload_mode == "File Upload":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Save the uploaded file locally
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Classify the uploaded audio
        result = classify_gender("temp_audio.wav")
        st.write(f"Audio Duration: {result['audio_length']}")
        st.write(f"Male Probability: {result['male_probability']}")
        st.write(f"Female Probability: {result['female_probability']}")
        st.success(f"Prediction: {result['gender']}")

st.write("[💡 By Elyor 😊](https://www.linkedin.com/in/elyordev)")