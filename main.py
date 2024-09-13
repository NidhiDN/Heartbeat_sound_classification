import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import seaborn as sns
from keras.src.saving.saving_api import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

labels = pd.read_csv(r"F:\DHD\labels.csv")
print(labels.head())
print(labels['label'].value_counts())

spectrograms_list = []
label_list = []
fixed_len = 1000

#preprocessing
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

#EDA Analysis
class_distribution = labels['label'].value_counts()
print(class_distribution)

plt.figure(figsize=(10, 5))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

#Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(spectrograms_list, label_list, test_size=0.2, stratify=label_list,
                                                    random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

spectrograms_array = np.stack(spectrograms_list)
X_train = np.stack(X_train)
X_val = np.stack(X_val)
X_test = np.stack(X_test)

print(f'Training set: {X_train.shape}, {len(y_train)}')
print(f'Validation set: {X_val.shape}, {len(y_val)}')
print(f'Test set: {X_test.shape}, {len(y_test)}')

#model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 1000, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


X_train = X_train.reshape(X_train.shape[0], 128, 1000)
X_val = X_val.reshape(X_val.shape[0], 128, 1000)
X_test = X_test.reshape(X_test.shape[0], 128, 1000)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)

history = model.fit(X_train[..., np.newaxis], y_train, validation_data=(X_val[..., np.newaxis], y_val),
                    epochs=10, batch_size=16)

all_labels = np.concatenate((y_train, y_test))

encoder = LabelEncoder()
encoder.fit(all_labels)

y_train_encoded = encoder.transform(y_train)
y_test_encoded = encoder.transform(y_test)

y_pred = model.predict(X_test[..., np.newaxis])
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

print(classification_report(y_test_encoded, y_pred_classes))
print(confusion_matrix(y_test_encoded, y_pred_classes))

model.save(r"F:\newmod-fin.h5")

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np


model =load_model(r"F:\newmod-fin.h5")
# Load the saved model
from keras.models import load_model

model = load_model(r"F:\newmod-fin.h5")

# Load test data
X_test = np.stack(X_test)
X_test = X_test.reshape(X_test.shape[0], 128, 1000, 1)

# Predict classes for test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Encode labels
encoder = LabelEncoder()
encoder.fit(y_test)
y_test_encoded = encoder.transform(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_classes)
print("Accuracy:", accuracy)


filename = r"F:\DHD\audio\\" + labels['filename'][1800]
print(filename)
waveform, sample_rate = torchaudio.load(filename)
plt.figure(figsize=(14, 5))
plt.plot(waveform[0].numpy())
plt.title('Waveplot - Heart Sound')
plt.show()
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)(waveform)
mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram_db[0].numpy(), aspect='auto', origin='lower', cmap='magma',
           extent=(0, waveform.size(1) / sample_rate, 0, 8000))
plt.title('Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(format='%+2.0f dB')
plt.show()

# Define the fixed length of the spectrogram
fixed_length = 1000  # Assuming this is defined somewhere in your code


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

