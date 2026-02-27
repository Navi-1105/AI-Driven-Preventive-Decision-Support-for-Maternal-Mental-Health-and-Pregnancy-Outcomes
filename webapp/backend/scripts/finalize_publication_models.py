import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Scalling>=10"
SCORE_COL = "Scalling"

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
EXPANDED_PREVENTIVE_FEATURES = STRICT_HISTORY_FEATURES + [
    "Current Appereance Acceptance",
    "Relationship with Mother in-law",
]
ENGINEERED_FEATURES = [
    "Age_x_Gravida",
    "Education_Gap",
    "Children_to_Age",
    "Miscarriage_x_Gravida",
    "Money_x_FamilySystem",
]
SCREENING_ASSISTED_ITEMS = [
    "Thoughts that you would be better off dead, or of hurting yourself",
    "Poor appetite or overeating",
    "Trouble falling or staying sleep or sleeping too much",
    "Moving or speaking so slowly that other people could have Noticed. ",
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
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _best_accuracy_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    # Minimal-change threshold tuning to improve accuracy without altering ROC-AUC.
    candidates = np.unique(np.round(y_proba, 4))
    if len(candidates) == 0:
        return 0.5
    best_t = 0.5
    best_acc = -1.0
    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return round(best_t, 4)


def _sha256sum(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
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


def _numeric_proxy(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if float(numeric.notna().mean()) >= 0.6:
        return numeric.astype(float)
    cat = pd.Categorical(series.astype(str).fillna("MISSING"))
    codes = pd.Series(cat.codes, index=series.index, dtype=float)
    return codes.mask(codes < 0)


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    age = _numeric_proxy(out["Age"])
    gravida = _numeric_proxy(out["Gravida"])
    female_edu = _numeric_proxy(out["Female Education"])
    husband_edu = _numeric_proxy(out["Husband Education"])
    total_children = _numeric_proxy(out["Total Number of Children"])
    prev_miscarriage = _numeric_proxy(out["Previous Miscarriage"])
    money = _numeric_proxy(out["Sufficient Money for Basic Needs"])
    family = _numeric_proxy(out["Family System"])

    out = out.assign(
        Age_x_Gravida=age * gravida,
        Education_Gap=husband_edu - female_edu,
        Children_to_Age=total_children / (age + 1.0),
        Miscarriage_x_Gravida=prev_miscarriage * gravida,
        Money_x_FamilySystem=money * family,
    )
    return out


def run(csv_path: Path, output_dir: Path, model_dir: Path):
    df = pd.read_csv(csv_path)
    # Explicitly remove any derived label column and build label from Scalling threshold.
    if "Labelling" in df.columns:
        df = df.drop(columns=["Labelling"])
    feature_set = EXPANDED_PREVENTIVE_FEATURES
    required_cols = feature_set + SCREENING_ASSISTED_ITEMS + [SCORE_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    df = df[required_cols].dropna(subset=[SCORE_COL])
    df = _add_engineered_features(df)
    model_features = feature_set + ENGINEERED_FEATURES
    screening_features = model_features + SCREENING_ASSISTED_ITEMS
    before_dedup_rows = len(df)
    df = df.drop_duplicates()
    after_dedup_rows = len(df)
    dropped_duplicate_rows = int(before_dedup_rows - after_dedup_rows)
    profile_groups = pd.util.hash_pandas_object(df[model_features].astype(str), index=False)
    unique_profiles = int(profile_groups.nunique())
    profile_duplicate_rows = int(len(df) - unique_profiles)
    score_numeric = pd.to_numeric(df[SCORE_COL], errors="coerce")
    near_threshold_rows = int(((score_numeric >= 9) & (score_numeric <= 11)).sum())
    temporal_columns = [
        c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()
    ]
    y = (pd.to_numeric(df[SCORE_COL], errors="coerce") >= 10).astype(int).to_numpy()
    X = df[model_features].copy()
    X_screening = df[screening_features].copy()
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_screening_train = X_screening.iloc[train_idx]
    X_screening_test = X_screening.iloc[test_idx]
    pre = _build_preprocessor(X_train)
    pre_lr = _build_preprocessor(X_train, scale_numeric=True)
    pre_screening = _build_preprocessor(X_screening_train)

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
    lr_pipe = Pipeline(
        [
            ("preprocess", pre_lr),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                CatBoostClassifier(
                    iterations=600,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=RANDOM_STATE,
                    auto_class_weights="Balanced",
                    verbose=False,
                ),
            ),
        ]
    )
    screening_pipe = Pipeline(
        [
            ("preprocess", pre_screening),
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    main_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    lr_pipe.fit(X_train, y_train)
    cat_pipe.fit(X_train, y_train)
    screening_pipe.fit(X_screening_train, y_train)
    calibrated_main = CalibratedClassifierCV(main_pipe, method="isotonic", cv=3)
    calibrated_main.fit(X_train, y_train)

    p_main_tr = main_pipe.predict_proba(X_train)[:, 1]
    p_main_te = main_pipe.predict_proba(X_test)[:, 1]
    p_main_te_cal = calibrated_main.predict_proba(X_test)[:, 1]
    p_rf_tr = rf_pipe.predict_proba(X_train)[:, 1]
    p_rf_te = rf_pipe.predict_proba(X_test)[:, 1]
    p_lr_tr = lr_pipe.predict_proba(X_train)[:, 1]
    p_lr_te = lr_pipe.predict_proba(X_test)[:, 1]
    p_cat_tr = cat_pipe.predict_proba(X_train)[:, 1]
    p_cat_te = cat_pipe.predict_proba(X_test)[:, 1]
    p_screen_tr = screening_pipe.predict_proba(X_screening_train)[:, 1]
    p_screen_te = screening_pipe.predict_proba(X_screening_test)[:, 1]
    t_main = _best_accuracy_threshold(y_train, p_main_tr)
    t_lr = _best_accuracy_threshold(y_train, p_lr_tr)
    t_cat = _best_accuracy_threshold(y_train, p_cat_tr)
    t_rf = _best_accuracy_threshold(y_train, p_rf_tr)
    t_screen = _best_accuracy_threshold(y_train, p_screen_tr)

    main_cv = cross_val_score(main_pipe, X, y, cv=cv, scoring="roc_auc")
    rf_cv = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc")
    lr_cv = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
    cat_cv = cross_val_score(cat_pipe, X, y, cv=cv, scoring="roc_auc")
    screening_cv = cross_val_score(screening_pipe, X_screening, y, cv=cv, scoring="roc_auc")
    groups = pd.util.hash_pandas_object(X.astype(str), index=False).to_numpy()
    group_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    main_cv_group = cross_val_score(main_pipe, X, y, cv=group_cv, groups=groups, scoring="roc_auc")
    rf_cv_group = cross_val_score(rf_pipe, X, y, cv=group_cv, groups=groups, scoring="roc_auc")
    lr_cv_group = cross_val_score(lr_pipe, X, y, cv=group_cv, groups=groups, scoring="roc_auc")
    cat_cv_group = cross_val_score(cat_pipe, X, y, cv=group_cv, groups=groups, scoring="roc_auc")
    groups_screening = pd.util.hash_pandas_object(X_screening.astype(str), index=False).to_numpy()
    screening_cv_group = cross_val_score(
        screening_pipe, X_screening, y, cv=group_cv, groups=groups_screening, scoring="roc_auc"
    )
    g_train_idx, g_test_idx = next(group_cv.split(X, y, groups))
    group_holdout = {}
    for key, pipe in (
        ("main_model", main_pipe),
        ("logistic_regression_baseline", lr_pipe),
        ("catboost_model", cat_pipe),
        ("sensitivity_model", rf_pipe),
    ):
        m = clone(pipe)
        m.fit(X.iloc[g_train_idx], y[g_train_idx])
        p = m.predict_proba(X.iloc[g_test_idx])[:, 1]
        group_holdout[key] = round(float(roc_auc_score(y[g_test_idx], p)), 4)
    sg_train_idx, sg_test_idx = next(group_cv.split(X_screening, y, groups_screening))
    screening_holdout_model = clone(screening_pipe)
    screening_holdout_model.fit(X_screening.iloc[sg_train_idx], y[sg_train_idx])
    screening_holdout_auc = round(
        float(roc_auc_score(y[sg_test_idx], screening_holdout_model.predict_proba(X_screening.iloc[sg_test_idx])[:, 1])),
        4,
    )

    payload = {
        "feature_set": model_features,
        "base_feature_set": feature_set,
        "engineered_features": ENGINEERED_FEATURES,
        "screening_assisted_items": SCREENING_ASSISTED_ITEMS,
        "dataset_rows": int(len(df)),
        "deduplication": {
            "before_rows": int(before_dedup_rows),
            "after_rows": int(after_dedup_rows),
            "dropped_exact_duplicate_rows": dropped_duplicate_rows,
            "unique_profiles_after_dedup": unique_profiles,
            "profile_duplicate_rows_after_dedup": profile_duplicate_rows,
        },
        "dataset_sha256": _sha256sum(csv_path),
        "label_definition": "y = (Scalling >= 10).astype(int)",
        "split_strategy": "train_test_split stratified 80/20 + StratifiedKFold(5) CV",
        "robustness_checks": {
            "grouped_cv_strategy": "StratifiedGroupKFold(5) on hashed feature profiles",
            "grouped_cv_auc_mean": {
                "main_model": round(float(np.mean(main_cv_group)), 4),
                "logistic_regression_baseline": round(float(np.mean(lr_cv_group)), 4),
                "catboost_model": round(float(np.mean(cat_cv_group)), 4),
                "sensitivity_model": round(float(np.mean(rf_cv_group)), 4),
                "screening_assisted_model": round(float(np.mean(screening_cv_group)), 4),
            },
            "grouped_cv_auc_std": {
                "main_model": round(float(np.std(main_cv_group)), 4),
                "logistic_regression_baseline": round(float(np.std(lr_cv_group)), 4),
                "catboost_model": round(float(np.std(cat_cv_group)), 4),
                "sensitivity_model": round(float(np.std(rf_cv_group)), 4),
                "screening_assisted_model": round(float(np.std(screening_cv_group)), 4),
            },
            "grouped_holdout_auc": group_holdout,
            "screening_assisted_grouped_holdout_auc": screening_holdout_auc,
            "grouped_split_profile_overlap_count": int(
                len(set(groups[g_train_idx]).intersection(set(groups[g_test_idx])))
            ),
            "label_quality_audit": {
                "positive_rate": round(float(np.mean(y)), 4),
                "positive_count": int(np.sum(y)),
                "negative_count": int(len(y) - np.sum(y)),
                "near_threshold_rows_9_to_11": near_threshold_rows,
                "score_min": float(score_numeric.min()),
                "score_max": float(score_numeric.max()),
            },
            "temporal_holdout_status": (
                "not_available_no_temporal_column" if not temporal_columns else "available"
            ),
            "temporal_columns_detected": temporal_columns,
        },
        "main_model": {
            "name": "lightgbm_preventive_main",
            "decision_threshold": t_main,
            "train_auc": round(float(roc_auc_score(y_train, p_main_tr)), 4),
            "cv_auc_mean": round(float(np.mean(main_cv)), 4),
            "cv_auc_std": round(float(np.std(main_cv)), 4),
            "test_metrics": _metrics(y_test, p_main_te, threshold=t_main),
        },
        "main_model_isotonic_calibrated": {
            "name": "lightgbm_preventive_main_isotonic",
            "test_metrics": _metrics(y_test, p_main_te_cal),
        },
        "logistic_regression_baseline": {
            "name": "logistic_regression_baseline",
            "decision_threshold": t_lr,
            "train_auc": round(float(roc_auc_score(y_train, p_lr_tr)), 4),
            "cv_auc_mean": round(float(np.mean(lr_cv)), 4),
            "cv_auc_std": round(float(np.std(lr_cv)), 4),
            "test_metrics": _metrics(y_test, p_lr_te, threshold=t_lr),
        },
        "catboost_model": {
            "name": "catboost_preventive_model",
            "decision_threshold": t_cat,
            "train_auc": round(float(roc_auc_score(y_train, p_cat_tr)), 4),
            "cv_auc_mean": round(float(np.mean(cat_cv)), 4),
            "cv_auc_std": round(float(np.std(cat_cv)), 4),
            "test_metrics": _metrics(y_test, p_cat_te, threshold=t_cat),
        },
        "screening_assisted_model": {
            "name": "lightgbm_screening_assisted",
            "decision_threshold": t_screen,
            "train_auc": round(float(roc_auc_score(y_train, p_screen_tr)), 4),
            "cv_auc_mean": round(float(np.mean(screening_cv)), 4),
            "cv_auc_std": round(float(np.std(screening_cv)), 4),
            "test_metrics": _metrics(y_test, p_screen_te, threshold=t_screen),
            "note": "Includes concurrent symptom items and is not preventive-only.",
        },
        "sensitivity_model": {
            "name": "random_forest_sensitivity",
            "decision_threshold": t_rf,
            "train_auc": round(float(roc_auc_score(y_train, p_rf_tr)), 4),
            "cv_auc_mean": round(float(np.mean(rf_cv)), 4),
            "cv_auc_std": round(float(np.std(rf_cv)), 4),
            "test_metrics": _metrics(y_test, p_rf_te, threshold=t_rf),
        },
        "governance_note": (
            "Main publication model is LightGBM on preventive history+psychosocial features with "
            "engineered interactions. Logistic Regression is reported as a linear baseline, "
            "CatBoost as an additional tabular benchmark, Random Forest as sensitivity, and a "
            "screening-assisted model is reported separately."
        ),
        "results_note": (
            "Tree-based ensemble models showed substantial training-set fit without proportional "
            "validation AUC gains, indicating overfitting risk in preventive-only prediction."
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
            "features": feature_set,
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
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Model & CV AUC & Test AUC & Recall & Precision & Brier \\\\",
        "\\hline",
    ]
    for key in (
        "logistic_regression_baseline",
        "main_model",
        "catboost_model",
        "sensitivity_model",
        "screening_assisted_model",
    ):
        row = payload[key]
        tm = row["test_metrics"]
        lines.append(
            f"{row['name'].replace('_', ' ').title()} & "
            f"{row['cv_auc_mean']:.3f} & {tm['roc_auc']:.3f} & {tm['recall']:.3f} & "
            f"{tm['precision']:.3f} & {tm['brier_score']:.3f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (output_dir / "tables" / "publication_model_selection.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Main model confusion matrix
    cm = np.array(payload["main_model"]["test_metrics"]["confusion_matrix"])
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title("LightGBM Main: Confusion Matrix (@0.5)")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_xticks([0, 1], labels=["Not", "Depressed"])
    ax_cm.set_yticks([0, 1], labels=["Not", "Depressed"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "publication_main_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cm)

    # Main model ROC curve
    fpr, tpr, _ = roc_curve(y_test, p_main_te)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, label=f"AUC = {payload['main_model']['test_metrics']['roc_auc']:.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("LightGBM Main: ROC Curve")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "publication_main_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig_roc)

    # Main model Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, p_main_te)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    ax_pr.plot(recall, precision, color="#2a9d8f")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("LightGBM Main: Precision-Recall Curve")
    ax_pr.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "publication_main_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig_pr)

    # Calibration curve (uncalibrated vs isotonic calibrated LightGBM).
    frac_pos_raw, mean_pred_raw = calibration_curve(y_test, p_main_te, n_bins=10, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, p_main_te_cal, n_bins=10, strategy="quantile")
    fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
    ax_cal.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax_cal.plot(mean_pred_raw, frac_pos_raw, marker="o", label="LightGBM raw")
    ax_cal.plot(mean_pred_cal, frac_pos_cal, marker="o", label="LightGBM isotonic")
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed positive fraction")
    ax_cal.set_title("Calibration Curve (Test Set)")
    ax_cal.legend()
    ax_cal.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "publication_calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cal)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["LightGBM Main", "LR Baseline", "CatBoost", "RF Sensitivity", "Screening-Assisted"]
    aucs = [
        payload["main_model"]["test_metrics"]["roc_auc"],
        payload["logistic_regression_baseline"]["test_metrics"]["roc_auc"],
        payload["catboost_model"]["test_metrics"]["roc_auc"],
        payload["sensitivity_model"]["test_metrics"]["roc_auc"],
        payload["screening_assisted_model"]["test_metrics"]["roc_auc"],
    ]
    ax.bar(names, aucs, color=["#2a9d8f", "#e9c46a", "#f4a261", "#457b9d", "#bc6c25"])
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
    print("Saved:", output_dir / "figures" / "publication_calibration_curve.png")
    print("Saved:", output_dir / "figures" / "publication_main_confusion_matrix.png")
    print("Saved:", output_dir / "figures" / "publication_main_roc_curve.png")
    print("Saved:", output_dir / "figures" / "publication_main_pr_curve.png")
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
