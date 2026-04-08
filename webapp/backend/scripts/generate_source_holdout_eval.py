import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import train_model_combined as t

SOURCES = ["main", "ppd", "uganda"]

PARAM_SPACE = {
    "xgb__n_estimators": [250, 300, 500, 700],
    "xgb__max_depth": [3, 4, 5, 6],
    "xgb__learning_rate": [0.02, 0.03, 0.05, 0.08],
    "xgb__subsample": [0.75, 0.85, 0.95],
    "xgb__colsample_bytree": [0.7, 0.85, 1.0],
    "xgb__min_child_weight": [1, 3, 5, 7],
    "xgb__reg_lambda": [1.0, 2.0, 4.0, 8.0],
    "xgb__reg_alpha": [0.0, 0.5, 1.0],
    "xgb__gamma": [0.0, 0.1, 0.2],
}


def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
    }


def _source_weights(source_series: pd.Series) -> np.ndarray:
    counts = source_series.value_counts().to_dict()
    weights = source_series.map(lambda s: 1.0 / counts[s]).astype(float).to_numpy()
    return weights / max(np.mean(weights), 1e-9)


def _stable_feature_subset(
    train_df: pd.DataFrame,
    base_features: list[str],
    max_missing_overall: float = 0.995,
) -> tuple[list[str], dict]:
    missing_by_source = (
        train_df.groupby(t.SOURCE)[base_features].apply(lambda d: d.isna().mean())
    )
    selected = []
    dropped = {}
    for col in base_features:
        overall_missing = float(train_df[col].isna().mean())
        non_missing = train_df[col].dropna()
        n_unique = int(non_missing.nunique()) if not non_missing.empty else 0
        if overall_missing > max_missing_overall:
            dropped[col] = {
                "reason": "near_all_missing_overall",
                "overall_missing": round(overall_missing, 4),
                "n_unique_non_missing": n_unique,
            }
            continue
        if n_unique <= 1:
            dropped[col] = {
                "reason": "non_informative_non_missing",
                "overall_missing": round(overall_missing, 4),
                "n_unique_non_missing": n_unique,
            }
            continue
        selected.append(col)
    if not selected:
        selected = base_features[:]
    report = {
        "selected_features": selected,
        "dropped_features": dropped,
        "missing_by_source": {
            src: {c: round(float(v), 4) for c, v in row.items()}
            for src, row in missing_by_source.to_dict(orient="index").items()
        },
        "rules": {
            "max_missing_overall": max_missing_overall,
            "note": "Cross-source missingness is handled via explicit missing indicators.",
        },
    }
    return selected, report


def _build_matrix(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    X = df[selected_features].copy()
    missing_cols = {
        f"{col}__missing": X[col].isna().astype(float).to_numpy() for col in selected_features
    }
    return pd.concat([X, pd.DataFrame(missing_cols, index=X.index)], axis=1)


def _sample_param_candidates(n_iter: int) -> list[dict]:
    rng = np.random.RandomState(t.RANDOM_STATE)
    candidates = []
    for _ in range(n_iter):
        params = {}
        for k, v in PARAM_SPACE.items():
            sampled = rng.choice(v)
            if isinstance(sampled, np.generic):
                sampled = sampled.item()
            params[k] = sampled
        candidates.append(params)
    return candidates


def _fit_base_pipeline(X_train, y_train, params: dict | None, sample_weight: np.ndarray):
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count) / max(1.0, float(pos_count))
    model = t._build_pipeline(scale_pos_weight=scale_pos_weight)
    if params:
        model.set_params(**params)
    model.fit(X_train, y_train, xgb__sample_weight=sample_weight)
    return model, scale_pos_weight


def _source_aware_objective(
    train_df: pd.DataFrame,
    selected_features: list[str],
    params: dict,
    recall_floor: float,
) -> dict:
    sources = sorted(train_df[t.SOURCE].unique().tolist())
    fold_rows = []
    for val_source in sources:
        tr_df = train_df[train_df[t.SOURCE] != val_source].copy()
        va_df = train_df[train_df[t.SOURCE] == val_source].copy()
        if tr_df.empty or va_df.empty:
            continue
        y_tr = tr_df[t.TARGET].astype(int).to_numpy()
        y_va = va_df[t.TARGET].astype(int).to_numpy()
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        X_tr = _build_matrix(tr_df, selected_features)
        X_va = _build_matrix(va_df, selected_features)
        weights = _source_weights(tr_df[t.SOURCE])
        model, _ = _fit_base_pipeline(X_tr, y_tr, params=params, sample_weight=weights)
        proba = model.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_va, proba))
        rec = float(recall_score(y_va, pred, zero_division=0))
        fold_rows.append({"val_source": val_source, "auc": auc, "recall": rec})

    if not fold_rows:
        return {
            "objective": -1.0,
            "mean_auc": 0.0,
            "min_recall": 0.0,
            "penalty": 1.0,
            "fold_rows": fold_rows,
        }

    aucs = [r["auc"] for r in fold_rows]
    recalls = [r["recall"] for r in fold_rows]
    mean_auc = float(np.mean(aucs))
    min_recall = float(np.min(recalls))
    penalty = max(0.0, recall_floor - min_recall)
    objective = mean_auc - 0.2 * penalty
    return {
        "objective": round(objective, 6),
        "mean_auc": round(mean_auc, 6),
        "min_recall": round(min_recall, 6),
        "penalty": round(penalty, 6),
        "fold_rows": fold_rows,
    }


def _fit_model(
    train_df: pd.DataFrame,
    selected_features: list[str],
    tune: bool,
    n_iter: int,
    recall_floor_for_selection: float,
):
    X_train = _build_matrix(train_df, selected_features)
    y_train = train_df[t.TARGET].astype(int).to_numpy()
    weights = _source_weights(train_df[t.SOURCE])

    if not tune:
        model, scale_pos_weight = _fit_base_pipeline(X_train, y_train, params=None, sample_weight=weights)
        return model, {
            "enabled": False,
            "strategy": "default_params",
            "scale_pos_weight": round(scale_pos_weight, 4),
        }

    best = None
    for params in _sample_param_candidates(n_iter):
        score = _source_aware_objective(
            train_df=train_df,
            selected_features=selected_features,
            params=params,
            recall_floor=recall_floor_for_selection,
        )
        row = {"params": params, **score}
        if best is None or row["objective"] > best["objective"]:
            best = row

    assert best is not None
    model, scale_pos_weight = _fit_base_pipeline(
        X_train,
        y_train,
        params=best["params"],
        sample_weight=weights,
    )
    return model, {
        "enabled": True,
        "n_iter": n_iter,
        "selection_objective": "mean_source_auc_with_recall_floor_penalty",
        "recall_floor_for_selection": recall_floor_for_selection,
        "best_objective": best["objective"],
        "best_mean_source_auc": best["mean_auc"],
        "best_min_source_recall": best["min_recall"],
        "best_params": best["params"],
        "best_fold_rows": best["fold_rows"],
        "scale_pos_weight": round(scale_pos_weight, 4),
        "source_weighting": "inverse_source_frequency",
    }


def _select_threshold_with_recall_floor(
    y_true: np.ndarray, y_proba: np.ndarray, min_recall: float
) -> dict:
    thresholds = np.linspace(0.1, 0.9, 161)
    candidates = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        pre = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        tn = float(cm[0, 0]) if cm.shape == (2, 2) else 0.0
        fp = float(cm[0, 1]) if cm.shape == (2, 2) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        row = {
            "threshold": float(threshold),
            "recall": rec,
            "precision": pre,
            "f1": f1,
            "accuracy": acc,
            "specificity": specificity,
        }
        if rec >= min_recall:
            candidates.append(row)

    if candidates:
        best = max(candidates, key=lambda x: (x["precision"], x["f1"], x["accuracy"]))
        return {
            "strategy": "recall_floor",
            "min_recall": float(min_recall),
            "threshold": round(best["threshold"], 4),
            "recall": round(best["recall"], 4),
            "precision": round(best["precision"], 4),
            "f1": round(best["f1"], 4),
            "accuracy": round(best["accuracy"], 4),
            "specificity": round(best["specificity"], 4),
            "fallback_used": False,
        }

    fallback = t._find_best_threshold(y_true, y_proba)
    return {
        "strategy": "recall_floor",
        "min_recall": float(min_recall),
        "threshold": float(fallback["threshold"]),
        "recall": float(fallback["recall"]),
        "precision": float(fallback["precision"]),
        "f1": float(fallback["f1"]),
        "accuracy": float(fallback["accuracy"]),
        "specificity": None,
        "fallback_used": True,
    }


def _threshold_from_validation(
    fitted_model,
    train_df: pd.DataFrame,
    selected_features: list[str],
    threshold_strategy: str,
    min_recall: float,
):
    X = _build_matrix(train_df, selected_features)
    y = train_df[t.TARGET].astype(int).to_numpy()
    src = train_df[t.SOURCE]

    X_tr, X_val, y_tr, y_val, src_tr, _ = train_test_split(
        X,
        y,
        src,
        test_size=0.2,
        random_state=t.RANDOM_STATE,
        stratify=y,
    )
    local = clone(fitted_model)
    local.fit(X_tr, y_tr, xgb__sample_weight=_source_weights(src_tr))
    val_proba = local.predict_proba(X_val)[:, 1]
    if threshold_strategy == "recall_floor":
        selected = _select_threshold_with_recall_floor(y_val, val_proba, min_recall=min_recall)
    else:
        selected = t._find_best_threshold(y_val, val_proba)
    return float(selected["threshold"]), selected


def run(
    output_dir: Path,
    tune: bool,
    n_iter: int,
    threshold_strategy: str,
    min_recall: float,
):
    root = t.ROOT_DIR
    df = t._assemble_dataset(
        root / "dataset.csv",
        root / "Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv",
        root / "Large Scale Anonymized EPDS Data for Prenatal Women in Selected hospitals in Uganda/records.csv",
        set(SOURCES),
    )
    df = df.drop_duplicates(subset=t.COMMON_FEATURES + [t.TARGET]).reset_index(drop=True)

    results = {}
    for holdout in SOURCES:
        train_df = df[df[t.SOURCE] != holdout].copy()
        test_df = df[df[t.SOURCE] == holdout].copy()

        train_df = train_df.drop_duplicates(subset=t.COMMON_FEATURES + [t.TARGET]).reset_index(drop=True)
        test_df = test_df.drop_duplicates(subset=t.COMMON_FEATURES + [t.TARGET]).reset_index(drop=True)

        y_train = train_df[t.TARGET].astype(int)
        y_test = test_df[t.TARGET].astype(int)
        train_neg = int((y_train == 0).sum())
        train_pos = int((y_train == 1).sum())
        train_imbalance_ratio = round(train_neg / max(1.0, float(train_pos)), 4)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            results[holdout] = {"error": "single_class_detected", "n_test": int(len(y_test))}
            continue

        selected_features, feature_report = _stable_feature_subset(train_df, t.COMMON_FEATURES)
        model, tuning = _fit_model(
            train_df=train_df,
            selected_features=selected_features,
            tune=tune,
            n_iter=n_iter,
            recall_floor_for_selection=min_recall,
        )

        threshold, threshold_info = _threshold_from_validation(
            fitted_model=model,
            train_df=train_df,
            selected_features=selected_features,
            threshold_strategy=threshold_strategy,
            min_recall=min_recall,
        )

        X_test = _build_matrix(test_df, selected_features)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test.to_numpy(), y_proba, threshold=threshold)

        results[holdout] = {
            "train_sources": sorted([s for s in SOURCES if s != holdout]),
            "test_source": holdout,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "class_balance_train": {"0": train_neg, "1": train_pos},
            "train_imbalance_ratio_neg_to_pos": train_imbalance_ratio,
            "feature_stability": feature_report,
            "threshold_selected_on_train_validation": threshold,
            "threshold_validation_metrics": threshold_info,
            "metrics": metrics,
            "tuning": tuning,
        }

    summary = {}
    valid = [v["metrics"] for v in results.values() if "metrics" in v]
    if valid:
        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]:
            vals = [x[metric_name] for x in valid]
            summary[metric_name + "_mean"] = round(float(np.mean(vals)), 4)
            summary[metric_name + "_std"] = round(float(np.std(vals)), 4)

    payload = {
        "protocol": {
            "type": "leave_one_source_out",
            "sources": SOURCES,
            "deduplicate_before_split": True,
            "threshold_tuned_on_train_validation_only": True,
            "threshold_strategy": threshold_strategy,
            "min_recall": float(min_recall) if threshold_strategy == "recall_floor" else None,
            "model_selection_target": "maximize_mean_source_auc_with_recall_penalty",
            "source_weighting": "inverse_source_frequency",
            "cross_source_feature_stability_filter": True,
            "tune": tune,
            "n_iter": n_iter if tune else 0,
        },
        "by_holdout_source": results,
        "macro_summary": summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "source_holdout_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Source-Holdout Evaluation (Train on two sources, test on one)}",
        "\\label{tab:source_holdout}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Holdout Source & Accuracy & Precision & Recall & F1 & ROC-AUC & Brier \\\\",
        "\\hline",
    ]
    for holdout in SOURCES:
        row = results.get(holdout, {})
        if "metrics" not in row:
            lines.append(f"{holdout.title()} & NA & NA & NA & NA & NA & NA \\\\")
            continue
        m = row["metrics"]
        lines.append(
            f"{holdout.title()} & {m['accuracy']:.3f} & {m['precision']:.3f} & {m['recall']:.3f} & "
            f"{m['f1']:.3f} & {m['roc_auc']:.3f} & {m['brier_score']:.3f} \\\\"
        )
    if summary:
        lines += [
            "\\hline",
            f"Mean$\\pm$SD & {summary['accuracy_mean']:.3f}$\\pm${summary['accuracy_std']:.3f} & "
            f"{summary['precision_mean']:.3f}$\\pm${summary['precision_std']:.3f} & "
            f"{summary['recall_mean']:.3f}$\\pm${summary['recall_std']:.3f} & "
            f"{summary['f1_mean']:.3f}$\\pm${summary['f1_std']:.3f} & "
            f"{summary['roc_auc_mean']:.3f}$\\pm${summary['roc_auc_std']:.3f} & "
            f"{summary['brier_score_mean']:.3f}$\\pm${summary['brier_score_std']:.3f} \\\\",
        ]
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

    (tables_dir / "source_holdout_evaluation.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", output_dir / "source_holdout_evaluation.json")
    print("Saved:", tables_dir / "source_holdout_evaluation.tex")


def main():
    parser = argparse.ArgumentParser(description="Generate source-holdout evaluation for pooled model.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=t.ROOT_DIR / "webapp" / "results",
        help="Directory to save source holdout outputs.",
    )
    parser.add_argument("--tune", action="store_true", help="Enable source-aware randomized search.")
    parser.add_argument("--n-iter", type=int, default=20, help="Randomized search iterations if --tune.")
    parser.add_argument(
        "--threshold-strategy",
        choices=["f1", "recall_floor"],
        default="recall_floor",
        help="How to select threshold on train-validation split.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.75,
        help="Recall floor used only when --threshold-strategy recall_floor.",
    )
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        tune=args.tune,
        n_iter=args.n_iter,
        threshold_strategy=args.threshold_strategy,
        min_recall=float(args.min_recall),
    )


if __name__ == "__main__":
    main()
