import os
import numpy as np
import pandas as pd
import librosa
from skimage.transform import resize
from tqdm import tqdm

# paths
METADATA_PATH = "data/processed/metadata_emodb.csv"

FEATURE_PATH = "data/processed/features_stft.npy"
LABEL_PATH = "data/processed/labels_stft.npy"

# STFT parameters (from paper)
SAMPLE_RATE = 16000
N_FFT = 256
HOP_LENGTH = 128
IMG_SIZE = 128


def extract_spectrogram(filepath):
    # load audio
    signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)

    # STFT
    stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)

    spectrogram = np.abs(stft)

    # log scale with numerical safety
    spectrogram = librosa.amplitude_to_db(spectrogram + 1e-10)

    # resize
    spectrogram = resize(spectrogram, (IMG_SIZE, IMG_SIZE))

    # normalize between 0 and 1
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

    return spectrogram


def build_features():

    df = pd.read_csv(METADATA_PATH)

    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        spec = extract_spectrogram(row["filepath"])

        features.append(spec)
        labels.append(row["label"])

    X = np.array(features)
    y = np.array(labels)

    # CNN expects channel dimension
    X = X[..., np.newaxis]

    os.makedirs("data/processed", exist_ok=True)

    np.save(FEATURE_PATH, X)
    np.save(LABEL_PATH, y)

    print("\nSaved features →", FEATURE_PATH)
    print("Saved labels →", LABEL_PATH)

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)


if __name__ == "__main__":
    build_features()