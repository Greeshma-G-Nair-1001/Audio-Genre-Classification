# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import joblib

# # # Load your trained Random Forest model and scaler (if applicable)
# # model = joblib.load('rf_audio_pop_model.pkl')  # Replace with the actual path to your model
# # scaler = joblib.load('scaler_pop.pkl')  # If you used scaling during training, load the scaler

# # # Function to create frequency features (this assumes you have frequency encoding for features like album, artist, genre)
# # def generate_frequency_features(data, column_name):
# #     # This function creates frequency features for categorical columns like 'album_name', 'artists', etc.
# #     # You would use the training data to get these frequencies, but for now, assume we pass the relevant data
# #     frequency = data[column_name].value_counts()
# #     return data[column_name].map(frequency).fillna(0)

# # # Streamlit UI
# # st.title("Track Popularity Prediction")
# # st.markdown("""
# #     **Objective**: Predict track popularity score based on various features like track name, artist, etc.
# # """)

# # # Collect user input
# # track_name = st.text_input("Enter Track Name")
# # artist = st.text_input("Enter Artist Name")
# # duration = st.number_input("Enter Track Duration (in seconds)", min_value=0)
# # explicit = st.selectbox("Explicit Content (1 for Yes, 0 for No)", [0, 1])
# # danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
# # energy = st.slider("Energy", 0.0, 1.0, 0.5)
# # key = st.slider("Key", 0, 11, 0)
# # loudness = st.slider("Loudness", -60.0, 0.0, -5.0)
# # mode = st.selectbox("Mode (1 for Major, 0 for Minor)", [0, 1])
# # speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
# # acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
# # instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
# # liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
# # valence = st.slider("Valence", 0.0, 1.0, 0.5)
# # tempo = st.number_input("Tempo (Beats per minute)", min_value=0)
# # time_signature = st.selectbox("Time Signature", [3, 4])  # Assuming time signature is either 3 or 4 for simplicity

# # # Prepare the features as a pandas DataFrame for prediction
# # input_data = pd.DataFrame({
# #     'duration_ms': [duration * 1000],  # Convert duration to milliseconds
# #     'explicit': [explicit],
# #     'danceability': [danceability],
# #     'energy': [energy],
# #     'key': [key],
# #     'loudness': [loudness],
# #     'mode': [mode],
# #     'speechiness': [speechiness],
# #     'acousticness': [acousticness],
# #     'instrumentalness': [instrumentalness],
# #     'liveness': [liveness],
# #     'valence': [valence],
# #     'tempo': [tempo],
# #     'time_signature': [time_signature],
# #     'track_name_freq': [0],  # Assuming 0 for simplicity; you will need to calculate this based on training data
# #     'track_id_freq': [0],  # Same for track_id_freq
# #     'artists_freq': [0],  # Frequency of the artist (will be calculated in the next step)
# #     'album_name_freq': [0],  # Frequency of the album
# #     'track_genre_freq': [0]  # Frequency of the genre
# # })
# # training_data = pd.read_excel(r"H:\IIT-M GUVI\Projects capstone\FinalProject1\training_data_pop.xlsx")
# # # Assuming frequency encoding was done on training data for 'artists', 'album_name', and 'track_genre'
# # # For prediction, we need to calculate these frequency-based features from your training data
# # # Example for frequency encoding (you need to calculate these based on training data):

# # # Example: Suppose you have a DataFrame `training_data` that includes the same columns as the input data
# # # You would replace these 0s with actual frequency values based on your training data
# # # Here, I assume these frequencies are calculated from the training dataset

# # # Generate frequency features for `artists`, `album_name`, and `track_genre`
# # input_data['artists_freq'] = generate_frequency_features(training_data, 'artists')
# # input_data['album_name_freq'] = generate_frequency_features(training_data, 'album_name')
# # input_data['track_genre_freq'] = generate_frequency_features(training_data, 'track_genre')

# # # If you used scaling during training, apply the same scaler to the input data
# # if scaler:
# #     input_data_scaled = scaler.transform(input_data)
# # else:
# #     input_data_scaled = input_data

# # # Predict popularity score using the loaded model
# # if st.button('Predict Popularity'):
# #     prediction = model.predict(input_data_scaled)
# #     st.write(f"Predicted Popularity Score: {prediction[0]:.2f}")




# import streamlit as st
# import pandas as pd
# import joblib

# # Load the trained model and scaler
# model = joblib.load('rf_audio_pop_model.pkl')  # Replace with your model path
# scaler = joblib.load('scaler_pop.pkl')  # If scaling was used during training, load the scaler

# # Load pre-computed frequency mappings
# artists_freq = joblib.load('artists_freq.pkl')
# album_name_freq = joblib.load('album_name_freq.pkl')
# track_genre_freq = joblib.load('track_genre_freq.pkl')

# # Define the required columns (same as used for training)
# required_columns = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
#                     'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
#                     'time_signature',  'track_id_freq', 'artists_freq', 'album_name_freq','track_name_freq', 'track_genre_freq']

# # def generate_frequency_features(data, column_name):
# #     # This function creates frequency features for categorical columns like 'album_name', 'artists', etc.
# #     # You would use the training data to get these frequencies, but for now, assume we pass the relevant data
# #     frequency = data[column_name].value_counts()
# #     return data[column_name].map(frequency).fillna(0)
# def generate_frequency_features(data, column_name, frequency_data=None):
#     # If no frequency data is passed, return 0 (or any placeholder value)
#     if frequency_data is None:
#         return data[column_name].map(lambda x: 0).fillna(0)  # You can set this to 0, or any other placeholder value
#     return data[column_name].map(frequency_data).fillna(0)


# # Streamlit UI to take user input
# st.title("Track Popularity Prediction")
# st.markdown("""
#     **Objective**: Predict the popularity score of a track based on various features such as track name, artist, etc.
# """)

# # Collect user input
# track_name = st.text_input("Enter Track Name")
# #artist = st.text_input("Enter Artist Name")
# duration = st.number_input("Enter Track Duration (in seconds)", min_value=0)
# explicit = st.selectbox("Explicit Content (1 for Yes, 0 for No)", [0, 1])
# danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
# energy = st.slider("Energy", 0.0, 1.0, 0.5)
# key = st.slider("Key", 0, 11, 0)
# loudness = st.slider("Loudness", -60.0, 0.0, -5.0)
# mode = st.selectbox("Mode (1 for Major, 0 for Minor)", [0, 1])
# speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
# acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
# instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
# liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
# valence = st.slider("Valence", 0.0, 1.0, 0.5)
# tempo = st.number_input("Tempo (Beats per minute)", min_value=0)
# time_signature = st.selectbox("Time Signature", [3, 4])  # Assuming time signature is either 3 or 4 for simplicity

# # Prepare the features as a pandas DataFrame for prediction
# input_data = pd.DataFrame({
#     'duration_ms': [duration * 1000],  # Convert duration to milliseconds
#     'explicit': [explicit],
#     'danceability': [danceability],
#     'energy': [energy],
#     'key': [key],
#     'loudness': [loudness],
#     'mode': [mode],
#     'speechiness': [speechiness],
#     'acousticness': [acousticness],
#     'instrumentalness': [instrumentalness],
#     'liveness': [liveness],
#     'valence': [valence],
#     'tempo': [tempo],
#     'time_signature': [time_signature],
#     'track_id_freq': [0],  # Same for track_id_freq
#     'artists_freq': [0],  # Add the artist name here
#     'album_name_freq': [0],  # Frequency of the album
#     'track_name_freq': [0],  # Assuming 0 for simplicity; you will need to calculate this based on training data
#     'track_genre_freq': [0]  # Frequency of the genre
# })

# # Ensure that the input_data columns match the training data columns (same order)
# input_data = input_data[required_columns]

# # # Generate frequency features for `artists`, `album_name`, and `track_genre`
# # input_data['artists_freq'] = generate_frequency_features(input_data, 'artists', artists_freq)
# # input_data['album_name_freq'] = generate_frequency_features(input_data, 'album_name', album_name_freq)
# # input_data['track_genre_freq'] = generate_frequency_features(input_data, 'track_genre', track_genre_freq)
# # Assuming you have already loaded the frequency data (like artists_freq, album_name_freq, etc.)
# input_data['artists_freq'] = generate_frequency_features(input_data, 'artists_freq', frequency_data=artists_freq)
# input_data['album_name_freq'] = generate_frequency_features(input_data, 'album_name_freq', frequency_data=album_name_freq)
# input_data['track_genre_freq'] = generate_frequency_features(input_data, 'track_genre_freq', frequency_data=track_genre_freq)

# # Apply scaling if required
# input_data_scaled = scaler.transform(input_data)

# # Predict the popularity score using the trained model
# if st.button('Predict Popularity'):
#     prediction = model.predict(input_data_scaled)
#     st.write(f"Predicted Popularity Score: {prediction[0]:.2f}")



import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('rf_audio_pop_model.pkl')  # Replace with your model path
scaler = joblib.load('scaler_pop.pkl')  # If scaling was used during training, load the scaler

# Load pre-computed frequency mappings
artists_freq = joblib.load('artists_freq.pkl')
album_name_freq = joblib.load('album_name_freq.pkl')
track_genre_freq = joblib.load('track_genre_freq.pkl')

# Define the required columns (same as used for training)
required_columns = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                    'time_signature', 'track_id_freq', 'artists_freq', 'album_name_freq','track_name_freq', 'track_genre_freq']

# Function to generate frequency features based on a given frequency mapping
def generate_frequency_features(data, column_name, frequency_data=None):
    if frequency_data is None:
        return data[column_name].map(lambda x: 0).fillna(0)  # Default to 0 if no frequency data
    return data[column_name].map(frequency_data).fillna(0)

# Streamlit UI to take user input
st.title("Track Popularity Prediction")
st.markdown("""
    **Objective**: Predict the popularity score of a track based on various features such as track name, artist, etc.
    Each input below allows you to specify the characteristics of a track.
""")

# Descriptions for each input field
track_name_desc = "Select the name of the track you want to predict the popularity for."
duration_desc = "Enter the total duration of the track (in seconds)."
explicit_desc = "Select whether the track contains explicit content (1 for Yes, 0 for No)."
danceability_desc = "How suitable the track is for dancing. (0 to 1, 1 being very danceable)."
energy_desc = "How energetic the track is. (0 to 1, 1 being very energetic)."
key_desc = "Select the musical key of the track (0 to 11)."
loudness_desc = "Enter the track's loudness in decibels (range: -60 to 0)."
mode_desc = "Select the mode of the track (1 for Major, 0 for Minor)."
speechiness_desc = "How much spoken words are present in the track. (0 to 1, 1 being spoken word)."
acousticness_desc = "How acoustic the track is. (0 to 1, 1 being very acoustic)."
instrumentalness_desc = "How instrumental the track is. (0 to 1, 1 being fully instrumental)."
liveness_desc = "How live the track sounds. (0 to 1, 1 being very live)."
valence_desc = "How positive or happy the track sounds. (0 to 1, 1 being very positive)."
tempo_desc = "Enter the tempo of the track in beats per minute (BPM)."
time_signature_desc = "Select the time signature (e.g., 3 for 3/4, 4 for 4/4)."

# Collect user input for track details
track_names = ["Track A", "Track B", "Track C"]  # Replace this with a dynamic list of track names if available
selected_track = st.selectbox("Select Track Name", track_names, help=track_name_desc)

# Collect other user inputs
duration = st.number_input("Track Duration (in seconds)", min_value=0, help=duration_desc)
explicit = st.selectbox("Explicit Content (1 for Yes, 0 for No)", [0, 1], help=explicit_desc)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5, help=danceability_desc)
energy = st.slider("Energy", 0.0, 1.0, 0.5, help=energy_desc)
key = st.slider("Key", 0, 11, 0, help=key_desc)
loudness = st.slider("Loudness", -60.0, 0.0, -5.0, help=loudness_desc)
mode = st.selectbox("Mode (1 for Major, 0 for Minor)", [0, 1], help=mode_desc)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5, help=speechiness_desc)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, help=acousticness_desc)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5, help=instrumentalness_desc)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5, help=liveness_desc)
valence = st.slider("Valence", 0.0, 1.0, 0.5, help=valence_desc)
tempo = st.number_input("Tempo (Beats per minute)", min_value=0, help=tempo_desc)
time_signature = st.selectbox("Time Signature", [3, 4], help=time_signature_desc)

# Prepare the features as a pandas DataFrame for prediction
input_data = pd.DataFrame({
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
    'track_id_freq': [0],  # Placeholder; you may use actual track ID if needed
    'artists_freq': [0],  # Placeholder for artist frequency; replace with actual values if available
    'album_name_freq': [0],  # Placeholder for album frequency; replace as needed
    'track_name_freq': [0],  # Placeholder; replace with actual track frequency if available
    'track_genre_freq': [0]  # Placeholder for genre frequency; replace as needed
})

# Ensure that the input_data columns match the training data columns (same order)
input_data = input_data[required_columns]

# Apply frequency features for `track_name`, `artists`, `album_name`, etc.
input_data['track_name_freq'] = generate_frequency_features(input_data, 'track_name_freq', frequency_data={selected_track: 1})

# Apply scaling if required
input_data_scaled = scaler.transform(input_data)

# Predict the popularity score using the trained model
if st.button('Predict Popularity'):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Popularity Score: {prediction[0]:.2f}")
