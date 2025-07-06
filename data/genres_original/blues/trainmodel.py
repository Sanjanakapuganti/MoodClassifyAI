import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the base directory where the dataset is expected to be located after extraction
base_dir = '/content/data/genres_original'
output_csv_path = 'features.csv' # Save to the current working directory

# --- Feature Extraction Code (Revised from previous failed attempts) ---
all_features = []
genres = []

try:
    genres = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not genres:
        print(f"No genre subdirectories found in {base_dir}. Please ensure your data is extracted to this location.")
except FileNotFoundError:
    print(f"Error: The directory {base_dir} was not found. Please ensure your data is extracted and located at this path.")
    genres = [] # Set genres to empty to prevent further errors

if genres:
    print(f"Found genres: {genres}")
    for genre in genres:
        genre_path = os.path.join(base_dir, genre)
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            # Check if the file is likely an audio file
            if not file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                # print(f"Skipping non-audio file: {file_path}") # Keep this quiet unless debugging
                continue

            try:
                # Load the audio file
                y, sr = librosa.load(file_path, duration=30)

                # Extract features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).mean(axis=1)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
                mel = librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1)

                # Concatenate features
                features = np.hstack([mfcc, chroma, mel])

                # Append data
                all_features.append([file, genre] + features.tolist())
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

# Check if any features were extracted before creating DataFrame
if not all_features:
    print("No features were extracted. Cannot proceed with model training.")
else:
    # Define column names
    mfcc_cols = [f'mfcc_{i}' for i in range(40)]
    chroma_cols = [f'chroma_{i}' for i in range(12)]
    # Determine the number of mel features dynamically
    num_mel_features = len(all_features[0]) - 2 - len(mfcc_cols) - len(chroma_cols)
    mel_cols = [f'mel_{i}' for i in range(num_mel_features)]

    cols = ['filename', 'genre'] + mfcc_cols + chroma_cols + mel_cols

    # Create DataFrame and save to CSV
    features_df = pd.DataFrame(all_features, columns=cols)
    features_df.to_csv(output_csv_path, index=False)
    print(f"\nFeatures extracted and saved to {output_csv_path}.")
    display(features_df.head())

    # --- Model Training Code (If features_df was created) ---
    try:
        # 1. Load the extracted features and genre labels (already in features_df)
        df = features_df # Use the dataframe created from feature extraction

        # 2. Separate the features from the target variable
        X = df.drop(['filename', 'genre'], axis=1)
        y = df['genre']

        # 3. Encode the genre labels into numerical format
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print("\nGenre labels encoded.")

        # 4. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        print("\nData split into training and testing sets.")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

        # 5. Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("\nFeatures standardized.")

        # 6. Choose and instantiate a classification model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("\nRandomForestClassifier instantiated.")

        # 7. Train the model
        print("Training the model...")
        model.fit(X_train_scaled, y_train)
        print("Model training complete.")

        # 8. Evaluate the model's performance
        print("\nEvaluating the model...")
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 9. Print the evaluation metrics
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

    except Exception as e:
        print(f"An error occurred during model training or evaluation: {e}")
