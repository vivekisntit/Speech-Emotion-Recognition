import os
import pandas as pd
from sklearn.model_selection import train_test_split


# dataset root directory
DATA_DIR = "data/raw"

# dataset paths
RAVDESS = os.path.join(DATA_DIR, "ravdess")
CREMA = os.path.join(DATA_DIR, "crema")
TESS = os.path.join(DATA_DIR, "tess")
SAVEE = os.path.join(DATA_DIR, "savee")


def parse_ravdess():

    rows = []

    for actor in os.listdir(RAVDESS):

        actor_dir = os.path.join(RAVDESS, actor)

        for file in os.listdir(actor_dir):

            parts = file.split("-")
            emotion_id = int(parts[2])

            emotion_map = {
                1: "neutral",
                2: "calm",
                3: "happy",
                4: "sad",
                5: "angry",
                6: "fear",
                7: "disgust",
                8: "surprise"
            }

            emotion = emotion_map[emotion_id]

            rows.append({
                "path": os.path.join(actor_dir, file),
                "emotion": emotion,
                "dataset": "ravdess"
            })

    return rows


def parse_crema():

    rows = []

    emotion_map = {
        "SAD": "sad",
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral"
    }

    for file in os.listdir(CREMA):

        parts = file.split("_")
        emotion = emotion_map.get(parts[2], None)

        if emotion is None:
            continue

        rows.append({
            "path": os.path.join(CREMA, file),
            "emotion": emotion,
            "dataset": "cremad"
        })

    return rows


def parse_tess():

    rows = []

    for folder in os.listdir(TESS):

        folder_path = os.path.join(TESS, folder)

        for file in os.listdir(folder_path):

            emotion = file.split("_")[2].split(".")[0]

            if emotion == "ps":
                emotion = "surprise"

            rows.append({
                "path": os.path.join(folder_path, file),
                "emotion": emotion,
                "dataset": "tess"
            })

    return rows


def parse_savee():

    rows = []

    emotion_map = {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprise"
    }

    for file in os.listdir(SAVEE):

        part = file.split("_")[1]
        code = part[:-6]

        emotion = emotion_map.get(code, None)

        if emotion is None:
            continue

        rows.append({
            "path": os.path.join(SAVEE, file),
            "emotion": emotion,
            "dataset": "savee"
        })

    return rows


def build_metadata():

    rows = []

    rows.extend(parse_ravdess())
    rows.extend(parse_crema())
    rows.extend(parse_tess())
    rows.extend(parse_savee())

    df = pd.DataFrame(rows)

    return df


def create_splits(df):

    train, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["emotion"],
        random_state=42
    )

    train, val = train_test_split(
        train,
        test_size=0.1,
        stratify=train["emotion"],
        random_state=42
    )

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    df = pd.concat([train, val, test])

    return df


def main():

    print("Building metadata...")

    df = build_metadata()

    print("Creating splits...")

    df = create_splits(df)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv("data/processed/metadata.csv", index=False)

    print("Saved metadata to data/processed/metadata.csv")

    print(df.head())


if __name__ == "__main__":
    main()