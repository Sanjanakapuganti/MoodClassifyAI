# MoodClassifyAI
AI model that  infers mood from audio using ML &amp; librosa
# 🎵 AI Music Genre & Mood Predictor

This project predicts the **genre** and **mood** of a given `.wav` audio file using machine learning and audio signal processing.

<img width="866" height="661" alt="Screenshot 2025-07-07 015655" src="https://github.com/user-attachments/assets/ef452ca8-205c-4108-8237-6f6a1273559c" />

📊 Model Evaluation Metrics

Our music genre classification model was trained to recognize 10 different genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock).

Overall Performance:

Accuracy: 64%

Precision: 65.6%

Recall: 64%

F1-score: 64.2%

Class-wise Performance:

Strong results in blues (0.84 F1), classical (0.83 F1), metal (0.76 F1), and pop (0.76 F1).

Moderate performance in disco (0.65 F1), hiphop (0.62 F1), and jazz (0.55 F1).

Lower accuracy for country (0.42 F1), reggae (0.33 F1), and rock (0.54 F1), indicating more confusion among these classes.

Interpretation:

The model performs well on genres with distinct musical features like blues, classical, and metal.

It struggles with genres that have overlapping patterns (e.g., country vs. rock, reggae vs. hiphop), which is expected in audio-based classification tasks.

A balanced dataset, feature engineering, or deep learning architectures (e.g., CNNs on spectrograms) could further improve classification.
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
