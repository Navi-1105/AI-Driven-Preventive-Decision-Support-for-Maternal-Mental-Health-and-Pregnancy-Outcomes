import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED_DATA = [
    {"text": "I gave birth recently and I feel hopeless and cannot sleep", "label": "high"},
    {"text": "I feel like hurting myself after delivery", "label": "high"},
    {"text": "I am exhausted and crying all day with my newborn", "label": "high"},
    {"text": "Mujhe bachche ke baad bahut udaasi aur neend ki dikkat hai", "label": "high"},
    {"text": "Delivery ke baad mujhe jeene ka mann nahi karta", "label": "high"},
    {"text": "I am worried and tired most days after birth", "label": "moderate"},
    {"text": "I feel anxious in my third trimester and cannot eat well", "label": "moderate"},
    {"text": "Pregnancy stress is affecting my sleep", "label": "moderate"},
    {"text": "Main pregnancy mein bahut tension me hoon aur neend kam aati hai", "label": "moderate"},
    {"text": "Bachche ke baad thakan aur chinta rehti hai", "label": "moderate"},
    {"text": "I feel supported by my family and coping well", "label": "low"},
    {"text": "Sleep is improving and I am managing stress", "label": "low"},
    {"text": "I am doing better with counseling and support", "label": "low"},
    {"text": "Meri family support karti hai aur main better feel kar rahi hoon", "label": "low"},
    {"text": "Doctor se baat karne ke baad mujhe acha lag raha hai", "label": "low"},
]

FACTOR_MAP = {
    "sleep_disturbance": ["cannot sleep", "neend", "insomnia"],
    "low_mood": ["hopeless", "udaasi", "depressed", "crying"],
    "anxiety": ["anxious", "tension", "worry", "chinta"],
    "fatigue": ["exhausted", "thakan", "tired"],
    "bonding_difficulty": ["cannot bond", "detached from baby"],
    "guilt_or_worthlessness": ["bad mother", "worthless", "failure"],
    "self_harm_ideation": ["hurting myself", "jeene ka mann nahi", "kill myself"],
}


def build_training_frame(csv_path: Path | None) -> pd.DataFrame:
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        required = {"text", "label"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Training CSV missing columns: {missing}")
        return df[["text", "label"]].dropna()

    return pd.DataFrame(SEED_DATA)


def train(csv_path: Path | None, out_path: Path, metrics_path: Path | None = None):
    df = build_training_frame(csv_path).copy()
    df["label"] = df["label"].astype(str)
    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=stratify
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=8000,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(str))
    y_pred = pipeline.predict(test_df["text"].astype(str))
    report = classification_report(
        test_df["label"].astype(str), y_pred, output_dict=True, zero_division=0
    )
    conf = confusion_matrix(test_df["label"].astype(str), y_pred).tolist()

    payload = {
        "pipeline": pipeline,
        "label_classes": list(pipeline.named_steps["clf"].classes_),
        "factor_map": FACTOR_MAP,
        "sample_size": int(df.shape[0]),
        "source": str(csv_path) if csv_path else "seed_data",
        "metrics": {
            "classification_report": report,
            "confusion_matrix": conf,
            "test_size": int(test_df.shape[0]),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload["metrics"], indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV with columns text,label",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("model/chat_risk_model.joblib"),
        help="Output path for chat model",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("model/metrics/chat_model_metrics.json"),
        help="Output path for chat model metrics json",
    )
    args = parser.parse_args()

    payload = train(args.csv, args.out, args.metrics_out)
    print(
        f"Chat model trained on {payload['sample_size']} records from {payload['source']}. Saved to {args.out}"
    )


if __name__ == "__main__":
    main()
