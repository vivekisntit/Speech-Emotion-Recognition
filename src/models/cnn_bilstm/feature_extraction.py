import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

METADATA = "data/processed/metadata.csv"

FEATURE_OUT = "data/processed/features_logmel.npy"
LABEL_OUT = "data/processed/labels_logmel.npy"

SR = 22050
DURATION = 4
SAMPLES = SR * DURATION

N_MELS = 128


def extract_logmel(path):

    signal, sr = librosa.load(path, sr=SR)

    if len(signal) < SAMPLES:
        pad = SAMPLES - len(signal)
        signal = np.pad(signal, (0, pad))

    else:
        signal = signal[:SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=N_MELS
    )

    logmel = librosa.power_to_db(mel)

    return logmel


def main():

    df = pd.read_csv(METADATA)

    X = []
    Y = []

    print("Extracting log-mel spectrograms")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        path = row["path"]
        emotion = row["emotion"]

        try:
            feature = extract_logmel(path)
        except:
            continue

        X.append(feature)
        Y.append(emotion)

    X = np.array(X)
    Y = np.array(Y)

    np.save(FEATURE_OUT, X)
    np.save(LABEL_OUT, Y)

    print("Saved features:", X.shape)
    print("Saved labels:", Y.shape)


if __name__ == "__main__":
    main()