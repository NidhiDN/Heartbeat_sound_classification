import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import subprocess

def load_model_safe(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model_safe(r"F:\newmod-fin.h5")

fixed_length = 1000
encoder = LabelEncoder()
label_array = ["normal", "artifact", "murmur", "extrasystole", "extrahls"]
encoder.fit(label_array)
sample_rate = 44100

def predict(audio):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )(audio)
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    if mel_spectrogram_db.shape[2] > fixed_length:
        mel_spectrogram_db = mel_spectrogram_db[:, :, :fixed_length]
    else:
        padding = fixed_length - mel_spectrogram_db.shape[2]
        mel_spectrogram_db = torch.nn.functional.pad(mel_spectrogram_db, (0, padding))
    mel_spectrogram_db = torch.nn.functional.interpolate(mel_spectrogram_db.unsqueeze(0), size=(128, 1000)).squeeze(0)
    prediction = model.predict(mel_spectrogram_db.numpy().reshape(1, 128, 1000, 1))
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]
def main():
    st.title("Heart Beat Sound Prediction")
    predicted_category = ""
    st.write("Upload an audio file")
    uploaded_file = st.file_uploader("Upload", type=["wav", "mp3"])
    if uploaded_file is not None:
        waveform, _ = torchaudio.load(uploaded_file)
        waveform_np = waveform.numpy().flatten()
        fig, ax = plt.subplots()
        ax.plot(waveform[0].numpy())
        ax.set_title('Waveform')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )(waveform)
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        fig, ax = plt.subplots()
        ax.imshow(mel_spectrogram_db[0].numpy(), cmap='inferno', origin='lower')
        ax.set_title('Mel Spectrogram')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        predicted_category = predict(waveform)

        st.write(f"Predicted Heart Sound Category: {predicted_category}")
        if st.button("click here for more information", key="view_label_info"):
            subprocess.Popen(["streamlit", "run", "new_window.py", "--", f"--predicted_category={predicted_category}"])

if __name__ == "__main__":
    main()
