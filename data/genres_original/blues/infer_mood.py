import librosa

def predict_mood(path):
    y, sr = librosa.load(path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = librosa.feature.rms(y=y).mean()

    # Safely convert tempo and energy to scalars
    tempo = tempo.item() if hasattr(tempo, 'item') else float(tempo)
    energy = energy.item() if hasattr(energy, 'item') else float(energy)

    print(f"Tempo: {tempo:.2f}, Energy: {energy:.4f}")

    if energy > 0.05 and tempo > 120:
        return 'energetic'
    elif energy < 0.03:
        return 'calm'
    else:
        return 'melancholy'

# Test
mood = predict_mood('/content/data/genres_original/jazz/jazz.00000.wav')
print("🎭 Predicted Mood:", mood)
