import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train_model_combined as t

SOURCES = ["main", "ppd", "uganda"]

FEATURE_SETS = {
    "full10": t.COMMON_FEATURES,
    "core7": [
        "age",
        "gestational_weeks",
        "prior_pregnancy_loss",
        "support_score",
        "sleep_score",
        "fatigue_score",
        "low_mood_score",
    ],
    "core3": ["age", "gestational_weeks", "prior_pregnancy_loss"],
}


def build_matrix(df: pd.DataFrame, features: list[str], add_missing_indicators: bool) -> pd.DataFrame:
    x = df[features].copy()
    if add_missing_indicators:
        for c in features:
            x[f"{c}__missing"] = x[c].isna().astype(float)
    return x


def select_threshold_recall_floor(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.75) -> dict:
    best = None
    fallback = None
    for th in np.linspace(0.1, 0.9, 161):
        pred = (y_prob >= th).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        pre = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        row = {
            "threshold": float(th),
            "recall": float(rec),
            "precision": float(pre),
            "f1": float(f1),
            "accuracy": float(acc),
        }
        if fallback is None or (row["f1"], row["accuracy"]) > (fallback["f1"], fallback["accuracy"]):
            fallback = row
        if rec >= min_recall:
            if best is None or (row["precision"], row["f1"], row["accuracy"]) > (
                best["precision"],
                best["f1"],
                best["accuracy"],
            ):
                best = row
    if best is None:
        out = fallback
        out["fallback_used"] = True
        return out
    best["fallback_used"] = False
    return best


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def model_factory(name: str, pos_weight: float):
    if name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=t.RANDOM_STATE,
                    ),
                ),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=t.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "xgb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=450,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_lambda=4.0,
                        reg_alpha=0.5,
                        gamma=0.1,
                        random_state=t.RANDOM_STATE,
                        eval_metric="logloss",
                        tree_method="hist",
                        scale_pos_weight=pos_weight,
                    ),
                ),
            ]
        )
    raise ValueError(name)


def run(output_path: Path):
    root = t.ROOT_DIR
    df = t._assemble_dataset(
        root / "dataset.csv",
        root / "Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv",
        root / "Large Scale Anonymized EPDS Data for Prenatal Women in Selected hospitals in Uganda/records.csv",
        set(SOURCES),
    )
    df = df.drop_duplicates(subset=t.COMMON_FEATURES + [t.TARGET]).reset_index(drop=True)

    experiments = []
    for feature_set_name, features in FEATURE_SETS.items():
        for add_missing_indicators in [False, True]:
            for model_name in ["logreg", "rf", "xgb"]:
                for calibrate in [False, True]:
                    config = {
                        "feature_set": feature_set_name,
                        "add_missing_indicators": add_missing_indicators,
                        "model": model_name,
                        "calibrate_sigmoid": calibrate,
                    }
                    fold_metrics = []
                    valid = True
                    for holdout in SOURCES:
                        train_df = df[df[t.SOURCE] != holdout].copy().reset_index(drop=True)
                        test_df = df[df[t.SOURCE] == holdout].copy().reset_index(drop=True)

                        y_train = train_df[t.TARGET].astype(int)
                        y_test = test_df[t.TARGET].astype(int)
                        if y_train.nunique() < 2 or y_test.nunique() < 2:
                            valid = False
                            break

                        # split training into fit/calibration-threshold sets
                        tr_df, val_df = train_test_split(
                            train_df,
                            test_size=0.2,
                            random_state=t.RANDOM_STATE,
                            stratify=y_train,
                        )

                        x_tr = build_matrix(tr_df, features, add_missing_indicators)
                        y_tr = tr_df[t.TARGET].astype(int).to_numpy()
                        x_val = build_matrix(val_df, features, add_missing_indicators)
                        y_val = val_df[t.TARGET].astype(int).to_numpy()
                        x_test = build_matrix(test_df, features, add_missing_indicators)
                        y_te = y_test.to_numpy()

                        neg = int((y_tr == 0).sum())
                        pos = int((y_tr == 1).sum())
                        pos_weight = neg / max(pos, 1)

                        model = model_factory(model_name, pos_weight=pos_weight)
                        model.fit(x_tr, y_tr)

                        eval_model = model
                        if calibrate:
                            eval_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
                            eval_model.fit(x_val, y_val)

                        val_prob = eval_model.predict_proba(x_val)[:, 1]
                        selected = select_threshold_recall_floor(y_val, val_prob, min_recall=0.75)
                        te_prob = eval_model.predict_proba(x_test)[:, 1]
                        m = metrics(y_te, te_prob, threshold=selected["threshold"])
                        m["holdout_source"] = holdout
                        m["threshold"] = selected["threshold"]
                        m["val_recall"] = selected["recall"]
                        fold_metrics.append(m)

                    if not valid:
                        continue
                    macro = {
                        "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
                        "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
                        "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
                        "roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
                        "roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
                    }
                    experiments.append({"config": config, "by_source": fold_metrics, "macro": macro})

    # objective: maximize mean AUC, then mean F1, with soft recall floor at 0.45 across holdouts
    def score(exp):
        min_recall = min(m["recall"] for m in exp["by_source"])
        penalty = max(0.0, 0.45 - min_recall)
        return (exp["macro"]["roc_auc_mean"] - 0.15 * penalty, exp["macro"]["f1_mean"])

    experiments_sorted = sorted(experiments, key=score, reverse=True)
    payload = {
        "n_experiments": len(experiments_sorted),
        "best": experiments_sorted[0] if experiments_sorted else None,
        "top5": experiments_sorted[:5],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    if experiments_sorted:
        best = experiments_sorted[0]
        print("Best config:", best["config"])
        print("Best macro:", best["macro"])


if __name__ == "__main__":
    run(t.ROOT_DIR / "webapp" / "results" / "source_holdout_model_search.json")
