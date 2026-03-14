import os
import pandas as pd

# paths
RAW_DATA_PATH = "data/raw/emodb"
OUTPUT_PATH = "data/processed/metadata_emodb.csv"

# emotion mapping from EMO-DB
EMOTION_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral"
}


def extract_emotion(filename):
    """
    Extract emotion code from EMO-DB filename.
    Example: 03a01Fa.wav → F
    """
    emotion_code = filename[5]
    return EMOTION_MAP.get(emotion_code)


def build_metadata():
    data = []

    for file in os.listdir(RAW_DATA_PATH):

        if not file.endswith(".wav"):
            continue

        emotion = extract_emotion(file)

        if emotion is None:
            continue

        filepath = os.path.join(RAW_DATA_PATH, file)

        data.append({
            "filepath": filepath,
            "emotion": emotion
        })

    df = pd.DataFrame(data)

    # create numeric labels
    df["label"] = df["emotion"].astype("category").cat.codes

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Metadata saved → {OUTPUT_PATH}")
    print(df.head())
    print("\nDataset size:", len(df))


if __name__ == "__main__":
    build_metadata()