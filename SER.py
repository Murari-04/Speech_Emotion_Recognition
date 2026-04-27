import os
import numpy as np
import librosa
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tkinter import Tk, filedialog, Label, Button
import joblib
import matplotlib.pyplot as plt
import io

# Feature Extraction
def extract_features(file_name):
    audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

# Generate and plot spectrogram
def plot_spectrogram(file_name):
    audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    plt.imshow(D, aspect='auto', cmap='inferno', origin='lower',
               extent=[0, len(audio_data) / sample_rate, 0, sample_rate / 2])
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

# Speech-to-text conversion
def speech_to_text(file_name):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file_name)
    with audio_file as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Speech not recognized"
    except sr.RequestError:
        return "Speech recognition service unavailable"

# Load dataset and extract features
def load_data():
    data = []
    emotions = []
    dataset_path = 'dataset/'  # Ensure this path is correct
    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            feature_vector = extract_features(os.path.join(dataset_path, file))
            data.append(feature_vector)
            emotion = int(file.split('-')[2])  # Adjust based on your filename format
            emotions.append(emotion)
    return np.array(data), np.array(emotions)

# Emotion labels corresponding to the encoded labels
emotion_labels = ["happy", "sad", "angry", "fearful", "surprised", "neutral"]

# Load and preprocess data
X, y = load_data()
if len(X) == 0 or len(y) == 0:
    raise ValueError("No data loaded. Please check your dataset directory.")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, 'random_forest.pkl')
joblib.dump(svm_model, 'svm.pkl')
joblib.dump(nb_model, 'naive_bayes.pkl')

# Evaluation
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))

# Predict emotion
def predict_emotion(file_name, model):
    features = extract_features(file_name).reshape(1, -1)
    prediction = model.predict(features)
    return emotion_labels[prediction[0]]  # Convert prediction index to label

# Live voice input
def live_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Please speak...")
        audio = recognizer.listen(source)
        print("Audio captured, processing...")
    return audio

def predict_live_emotion():
    audio_data = live_voice_input()
    audio_stream = io.BytesIO(audio_data.get_wav_data())
    features = extract_features(audio_stream).reshape(1, -1)
    model = joblib.load('random_forest.pkl')  # You can switch models
    prediction = model.predict(features)
    emotion = emotion_labels[prediction[0]]

    # Convert speech to text
    transcription = speech_to_text(io.BytesIO(audio_data.get_wav_data()))

    # Update the GUI with results
    label_result.config(text=f"Predicted Emotion: {emotion}\nTranscribed Text: {transcription}")

# Tkinter GUI with speech-to-text
def load_model_and_predict():
    file_path = filedialog.askopenfilename()
    
    # Predict emotion
    model = joblib.load('random_forest.pkl')  # You can switch models
    emotion = predict_emotion(file_path, model)
    
    # Get the speech-to-text result
    transcription = speech_to_text(file_path)
    
    # Update the GUI with results
    label_result.config(text=f"Predicted Emotion: {emotion}\nTranscribed Text: {transcription}")
    
    # Plot spectrogram
    plot_spectrogram(file_path)

# Tkinter GUI
root = Tk()
root.title('Speech Emotion Recognition & Speech-to-Text')

label_result = Label(root, text="Upload an audio file or use live voice input", font=("Helvetica", 16))
label_result.pack()

btn_upload = Button(root, text="Upload Audio", command=load_model_and_predict)
btn_upload.pack()

btn_live_voice = Button(root, text="Live Voice Input", command=predict_live_emotion)
btn_live_voice.pack()

root.mainloop()
