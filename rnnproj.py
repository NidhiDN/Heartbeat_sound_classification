import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

labels = pd.read_csv(r"F:\DHD\labels.csv")
print(labels.head())
print(labels['label'].value_counts())

spectrograms_list = []
batch_size = 1000
label_list = []
fixed_len = 1000

for index, row in labels.iterrows():
    filename = r"F:\DHD\audio\\" + row['filename']
    waveform, sample_rate = torchaudio.load(filename, normalize=True)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_spectrogram_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )

    spectogram = mel_spectrogram_transform(waveform)
    spectogram = transforms.AmplitudeToDB()(spectogram)

    if spectogram.shape[2] > fixed_len:
        spectogram = spectogram[:, :, :fixed_len]
    elif spectogram.shape[2] < fixed_len:
        padding = fixed_len - spectogram.shape[2]
        pad = torch.zeros(spectogram.shape[0], spectogram.shape[1], padding)
        spectogram = torch.cat((spectogram, pad), dim=2)

    spectrograms_list.append(spectogram)
    label_list.append(row['label'])

spectrograms_array = np.stack(spectrograms_list)
label_array = np.array(label_list)

print(spectrograms_array.shape)
print(label_array.shape)

label_encoder = LabelEncoder()
label_array_encoded = label_encoder.fit_transform(label_array)

X_train, X_test, y_train, y_test = train_test_split(spectrograms_array, label_array_encoded, test_size=0.2, stratify=label_array_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], X_train.shape[2])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[3], X_val.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[3], X_test.shape[2])
print("Original shapes:")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
num_features = X_train.shape[2]
print("Number of features in X_train:", num_features)


from keras.initializers import Orthogonal

model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_initializer=Orthogonal()))
model.add(Dropout(0.5))
model.add(LSTM(units=64, return_sequences=True, kernel_initializer=Orthogonal()))
model.add(Dropout(0.5))
model.add(LSTM(units=32, kernel_initializer=Orthogonal()))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=5, activation='softmax'))


model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

model.save(r"F:\modddrnn-fin.h5")

model_path = r"F:\modddrnn-fin.h5"
model = load_model(model_path)

fixed_length=1000
def predict(audio_file):
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )(waveform)

        if mel_spectrogram.shape[2] > fixed_length:
            mel_spectrogram = mel_spectrogram[:, :, :fixed_length]
        else:
            padding = fixed_length - mel_spectrogram.shape[2]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))

        mel_spectrogram = torch.nn.functional.interpolate(mel_spectrogram.unsqueeze(0), size=(128, 1000)).squeeze(0)

        mel_spectrogram = mel_spectrogram.numpy().T[np.newaxis, :, :]

        prediction = model.predict(mel_spectrogram)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

        return predicted_class[0]
    except Exception as e:
        print(f"Error predicting for file: {audio_file}")
        print(e)
        return None

filename = r"F:\DHD\audio\\" + labels['filename'][7]
print(filename)
print(predict(filename))

true_labels_numeric = label_encoder.transform(labels['label'])

min_length = min(len(true_labels_numeric), len(y_pred_classes))
true_labels_numeric = true_labels_numeric[:min_length]
y_pred_classes = y_pred_classes[:min_length]

accuracy = accuracy_score(true_labels_numeric, y_pred_classes)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(true_labels_numeric, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)
