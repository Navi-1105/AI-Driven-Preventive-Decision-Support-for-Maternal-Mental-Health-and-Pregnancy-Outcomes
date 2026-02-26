import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TARGET = "Labelling"

STRICT_HISTORY_FEATURES = [
    "Age",
    "Gestational Age",
    "Number of sons ",
    "Number of daughters",
    "Total Number of Children",
    "Gravida",
    "Female Education",
    "Husband Education",
    "Working Status",
    "Physical Health ",
    "Previous Miscarriage",
    "Sufficient Money for Basic Needs",
    "Family System",
    "Male Gender Preference",
]


def _metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
    }


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def run(csv_path: Path, output_dir: Path, model_dir: Path):
    df = pd.read_csv(csv_path)
    df = df[STRICT_HISTORY_FEATURES + [TARGET]].dropna(subset=[TARGET])
    y = df[TARGET].astype(str).str.strip().str.lower().eq("depressed").astype(int).to_numpy()
    X = df[STRICT_HISTORY_FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pre = _build_preprocessor(X_train)

    main_pipe = Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    verbose=-1,
                ),
            ),
        ]
    )
    rf_pipe = Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    main_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    p_main_tr = main_pipe.predict_proba(X_train)[:, 1]
    p_main_te = main_pipe.predict_proba(X_test)[:, 1]
    p_rf_tr = rf_pipe.predict_proba(X_train)[:, 1]
    p_rf_te = rf_pipe.predict_proba(X_test)[:, 1]

    main_cv = cross_val_score(main_pipe, X, y, cv=cv, scoring="roc_auc")
    rf_cv = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc")

    payload = {
        "feature_set": STRICT_HISTORY_FEATURES,
        "dataset_rows": int(len(df)),
        "main_model": {
            "name": "lightgbm_preventive_main",
            "train_auc": round(float(roc_auc_score(y_train, p_main_tr)), 4),
            "cv_auc_mean": round(float(np.mean(main_cv)), 4),
            "cv_auc_std": round(float(np.std(main_cv)), 4),
            "test_metrics": _metrics(y_test, p_main_te),
        },
        "sensitivity_model": {
            "name": "random_forest_sensitivity",
            "train_auc": round(float(roc_auc_score(y_train, p_rf_tr)), 4),
            "cv_auc_mean": round(float(np.mean(rf_cv)), 4),
            "cv_auc_std": round(float(np.std(rf_cv)), 4),
            "test_metrics": _metrics(y_test, p_rf_te),
        },
        "governance_note": (
            "Main publication model is LightGBM on strict preventive history features. "
            "Random Forest retained as sensitivity analysis only."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "publication_model_selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    joblib.dump(
        {
            "pipeline": main_pipe,
            "features": STRICT_HISTORY_FEATURES,
            "target": TARGET,
            "label_classes": ["Not", "Depressed"],
            "metrics": payload["main_model"],
        },
        model_dir / "risk_model_preventive_main_lgbm.joblib",
    )

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Publication Model Selection (Preventive Setting)}",
        "\\label{tab:publication_model_selection}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Model & Train AUC & CV AUC & Test AUC & Recall & Precision & Brier \\\\",
        "\\hline",
    ]
    for key in ("main_model", "sensitivity_model"):
        row = payload[key]
        tm = row["test_metrics"]
        lines.append(
            f"{row['name'].replace('_', ' ').title()} & {row['train_auc']:.3f} & "
            f"{row['cv_auc_mean']:.3f} & {tm['roc_auc']:.3f} & {tm['recall']:.3f} & "
            f"{tm['precision']:.3f} & {tm['brier_score']:.3f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (output_dir / "tables" / "publication_model_selection.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["LightGBM Main", "RF Sensitivity"]
    aucs = [payload["main_model"]["test_metrics"]["roc_auc"], payload["sensitivity_model"]["test_metrics"]["roc_auc"]]
    ax.bar(names, aucs, color=["#2a9d8f", "#457b9d"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Publication Model vs Sensitivity Model")
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "publication_model_selection_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_dir / "publication_model_selection.json")
    print("Saved:", output_dir / "tables" / "publication_model_selection.tex")
    print("Saved:", output_dir / "figures" / "publication_model_selection_auc.png")
    print("Saved:", model_dir / "risk_model_preventive_main_lgbm.joblib")
    print("Main model test AUC:", payload["main_model"]["test_metrics"]["roc_auc"])


def main():
    parser = argparse.ArgumentParser(description="Finalize publication-safe model selection.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.csv, args.output_dir, args.model_dir)


if __name__ == "__main__":
    main()
