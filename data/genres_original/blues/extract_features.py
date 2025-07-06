import os
import librosa
import numpy as np
import pandas as pd

# Define the base directory where the dataset is located
base_dir = '/content/data/genres_original'

# Get a list of genre subdirectories within the base data directory
genres = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Initialize an empty list to store the extracted features and labels
all_features = []

# Loop through each genre directory
for genre in genres:
    genre_path = os.path.join(base_dir, genre)
    # Within each genre directory, loop through each audio file
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        try:
            # Load the audio file using librosa.load(), ensuring a duration is specified
            y, sr = librosa.load(file_path, duration=30)

            # Extract the following features from the loaded audio:
            # MFCCs (Mel-Frequency Cepstral Coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).mean(axis=1)
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
            # Mel spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1)

            # Concatenate the extracted mean features
            features = np.hstack([mfcc, chroma, mel])

            # Append the filename, genre label, and the feature vector
            all_features.append([file, genre] + features.tolist())
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

# Define the column names for the DataFrame
mfcc_cols = [f'mfcc_{i}' for i in range(40)]
chroma_cols = [f'chroma_{i}' for i in range(12)]
mel_cols = [f'mel_{i}' for i in range(len(mel))] # Use the last computed mel feature length
cols = ['filename', 'genre'] + mfcc_cols + chroma_cols + mel_cols

# Convert the list of features and labels into a Pandas DataFrame
features_df = pd.DataFrame(all_features, columns=cols)

# Save the DataFrame to a CSV file
features_df.to_csv('features.csv', index=False)

# Display the first few rows of the DataFrame
print(features_df.head())
