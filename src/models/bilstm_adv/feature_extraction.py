import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

METADATA = "data/processed/metadata.csv"
FEATURE_OUT = "data/processed/features_bilstm_adv.npy"
LABEL_OUT = "data/processed/labels_bilstm_adv.npy"

SR = 22050
DURATION = 4
SAMPLES = SR * DURATION
N_MELS = 128

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def extract_logmel(signal, sr):
    if len(signal) < SAMPLES:
        pad = SAMPLES - len(signal)
        signal = np.pad(signal, (0, pad))
    else:
        signal = signal[:SAMPLES]

    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel)
    return logmel

def main():
    df = pd.read_csv(METADATA)
    
    # Drop the imbalanced classes
    df = df[~df["emotion"].isin(["calm", "surprise"])]

    X = []
    Y = []

    print("Extracting and augmenting log-mel spectrograms...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["path"]
        emotion = row["emotion"]

        try:
            signal, sr = librosa.load(path, sr=SR)
            X.append(extract_logmel(signal, sr))
            Y.append(emotion)
            
            noise_signal = noise(signal)
            X.append(extract_logmel(noise_signal, sr))
            Y.append(emotion)

            stretch_signal = stretch(signal)
            pitch_signal = pitch(stretch_signal, sr)
            X.append(extract_logmel(pitch_signal, sr))
            Y.append(emotion)

        except Exception as e:
            continue

    X = np.array(X)
    Y = np.array(Y)

    np.save(FEATURE_OUT, X)
    np.save(LABEL_OUT, Y)

    print("Saved features:", X.shape)
    print("Saved labels:", Y.shape)

if __name__ == "__main__":
    main()