import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm


METADATA = "data/processed/metadata.csv"
OUTPUT = "data/processed/features_cnn_features.csv"


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def extract_features(data, sr):

    result = np.array([])

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))

    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))

    return result


def get_features(path):

    data, sr = librosa.load(path, duration=2.5, offset=0.6)

    result = []

    # original
    result.append(extract_features(data, sr))

    # noise
    noise_data = noise(data)
    result.append(extract_features(noise_data, sr))

    # stretch + pitch
    stretch_data = stretch(data)
    pitch_data = pitch(stretch_data, sr)
    result.append(extract_features(pitch_data, sr))

    return result


def main():

    df = pd.read_csv(METADATA)

    X = []
    Y = []

    print("Extracting features...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        path = row["path"]
        emotion = row["emotion"]

        try:
            features = get_features(path)
        except Exception as e:
            print("ERROR processing:", path)
            print(e)
            continue

        for feat in features:
            X.append(feat)
            Y.append(emotion)

    features_df = pd.DataFrame(X)
    features_df["label"] = Y

    features_df.to_csv(OUTPUT, index=False)

    print("Saved features to:", OUTPUT)
    print("Shape:", features_df.shape)


if __name__ == "__main__":
    main()