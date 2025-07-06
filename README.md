# MoodClassifyAI
AI model that  infers mood from audio using ML &amp; librosa
# 🎵 AI Music Genre & Mood Predictor

This project predicts the **genre** and **mood** of a given `.wav` audio file using machine learning and audio signal processing.

## 🔧 Technologies
- Python + Scikit-learn
- Librosa for audio feature extraction
- Random Forest for classification
- (Optional) DALL·E for mood-based art generation

## 📁 Structure
- `train.py`: Trains the model on `features.csv`
- `predict.py`: Predicts  mood from `.wav`
- `extract_features.py`: Extracts chroma, MFCC, tempo, etc.
- `art_generator.py`: Generates mood-based artwork

## 📦 Usage
```bash
python train.py
python predict.py --input example_audio/pop.00001.wav
Example output:

🎭 Mood: calm
🎨 Image: [Link to generated image] using AI
