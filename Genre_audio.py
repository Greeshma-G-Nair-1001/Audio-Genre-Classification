import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model for genre classification
model = joblib.load('rf_model_genre.pkl')  # Replace with your model path
scaler = joblib.load('scaler_clust.pkl')  # If scaling was used during training, load the scaler
encoder_artists = joblib.load('encoder_artists.pkl')  # Label encoder for artists (if used)
encoder_album = joblib.load('encoder_album.pkl')  # Label encoder for album_name (if used)
encoder_track_name = joblib.load('encoder_track_name.pkl')  # Label encoder for track_name (if used)

# Define the required columns (same as used for training, minus target column 'track_genre')
required_columns = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 
                    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 
                    'tempo', 'time_signature', 'artists', 'album_name', 'track_name']  # Don't include 'Unnamed: 0', 'track_id', 'track_genre'

# Streamlit UI to take user input
st.title("Track Genre Classification")
st.markdown("""**Objective**: Predict the genre of a track based on various audio features.""")

# Collect user input for track details
track_names = ["Track A", "Track B", "Track C"]  # Replace with dynamic track names if available
selected_track = st.selectbox("Select Track Name", track_names)

# Collect other user inputs
duration = st.number_input("Track Duration (in seconds)", min_value=0)
explicit = st.selectbox("Explicit Content (1 for Yes, 0 for No)", [0, 1])
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
key = st.slider("Key", 0, 11, 0)
loudness = st.slider("Loudness", -60.0, 0.0, -5.0)
mode = st.selectbox("Mode (1 for Major, 0 for Minor)", [0, 1])
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.number_input("Tempo (Beats per minute)", min_value=0)
time_signature = st.selectbox("Time Signature", [3, 4])
popularity = st.number_input("Track Popularity", min_value=0, max_value=100)  # Popularity input

# Collect artist and album name for encoding
selected_artist = st.text_input("Artist Name")
selected_album = st.text_input("Album Name")

# Prepare the features as a pandas DataFrame for prediction
input_data = pd.DataFrame({
    'popularity': [popularity],
    'duration_ms': [duration * 1000],  # Convert duration to milliseconds
    'explicit': [explicit],
    'danceability': [danceability],
    'energy': [energy],
    'key': [key],
    'loudness': [loudness],
    'mode': [mode],
    'speechiness': [speechiness],
    'acousticness': [acousticness],
    'instrumentalness': [instrumentalness],
    'liveness': [liveness],
    'valence': [valence],
    'tempo': [tempo],
    'time_signature': [time_signature],
    'artists': encoder_artists.transform([selected_artist])[0],  # Encode the artist name
    'album_name': encoder_album.transform([selected_album])[0],  # Encode the album name
    'track_name': encoder_track_name.transform([selected_track])[0]  # Encode the track name
})

# Ensure the input data has the same columns in the same order as used during training
input_data = input_data[required_columns]

# Apply scaling if required
input_data = input_data.astype('float32')  # Convert to float32 to reduce memory consumption
input_data_scaled = scaler.transform(input_data)

# Predict the genre using the trained model
if st.button('Predict Genre'):
    prediction = model.predict(input_data_scaled)
    predicted_genre = prediction[0]  # Get the predicted genre
    st.write(f"Predicted Genre: {predicted_genre}")
