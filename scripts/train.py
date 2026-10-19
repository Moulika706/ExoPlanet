"""Train and evaluate the ExoVet models (Phase 2).

For every model in the config the script:

1. splits the cleaned KOI table into train / calibration / test folds,
   grouped by host star so sibling KOIs never straddle the split;
2. fits the model on the train fold only;
3. calibrates it on the calibration fold (isotonic by default);
4. fits the split-conformal predictor on the calibrated calibration fold;
5. reports metrics on the untouched test fold.

The best model by calibrated macro-F1 is written to ``results/`` together
with a JSON metric bundle and a Markdown summary.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exovet.data import TARGET, clean_labels, load_config, load_raw
from exovet.explain import global_importance, transformed_feature_names
from exovet.models import MODEL_NAMES
from exovet.pipeline import VettingPipeline

RESULTS_DIR = Path("results")


def prepare_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the raw CSV, clean it and cache the processed table."""
    raw_path = config["data"]["raw_csv"]
    processed_path = Path(config["data"]["processed_csv"])
    df = clean_labels(load_raw(raw_path))
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"data: {len(df)} rows, {df[TARGET].nunique()} classes -> {processed_path}")
    return df


def format_metric_table(rows: List[Dict[str, Any]]) -> str:
    """Render the model comparison table."""
    header = (
        "| model | macro-F1 | PR-AUC | log-loss | Brier | ECE | "
        "recall@99% | conformal cov. | mean set |"
    )
    divider = "|" + "---|" * 9
    lines = [header, divider]
    for row in rows:
        calibrated = row["calibrated"]
        conformal = row["conformal"]
        lines.append(
            "| {model} | {f1:.4f} | {pr:.4f} | {ll:.4f} | {br:.4f} | {ece:.4f} | "
            "{r99:.4f} | {cov:.4f} | {size:.2f} |".format(
                model=row["model"],
                f1=calibrated["macro_f1"],
                pr=calibrated["macro_pr_auc"],
                ll=calibrated["log_loss"],
                br=calibrated["brier_score"],
                ece=calibrated["expected_calibration_error"],
                r99=calibrated["macro_recall_at_precision"],
                cov=conformal["empirical_coverage"],
                size=conformal["mean_set_size"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    seed = config.get("seed", 42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_data(config)
    eval_config = config.get("eval", {})
    splits_config = config.get("splits", {})
    conformal_config = config.get("conformal", {})
    shap_config = config.get("shap", {})

    models = list(MODEL_NAMES)
    results: List[Dict[str, Any]] = []
    fitted: Dict[str, VettingPipeline] = {}

    for model_name in models:
        print(f"\n=== {model_name} ===")
        started = time.perf_counter()
        pipeline = VettingPipeline(
            model_name=model_name,
            params=config["models"].get(model_name, {}),
            seed=seed,
            calibration_method=config["calibration"]["method"],
            conformal_alpha=conformal_config.get("alpha", 0.10),
            conformal_mode=conformal_config.get("mode", "marginal"),
            always_include_argmax=conformal_config.get("always_include_argmax", True),
            shap_perturbation=shap_config.get("perturbation", "tree_path"),
            top_k=config["report"]["top_k"],
        )
        pipeline.fit(
            df,
            n_splits=splits_config.get("n_folds", 5),
            test_fold=eval_config.get("test_fold", 0),
            calibration_fold=eval_config.get("calibration_fold", 1),
        )
        elapsed = time.perf_counter() - started
        metrics = pipeline.metrics
        calibrated = metrics["calibrated"]
        print(
            "  macro-F1 {macro_f1:.4f} | PR-AUC {macro_pr_auc:.4f} | "
            "log-loss {log_loss:.4f} | Brier {brier_score:.4f} | "
            "ECE {expected_calibration_error:.4f}".format(**calibrated)
        )
        print(
            "  conformal: coverage {empirical_coverage:.4f} (target {target_coverage:.2f}), "
            "mean set size {mean_set_size:.2f}, singleton rate {singleton_rate:.4f}".format(
                **metrics["conformal"]
            )
        )
        print(
            "  per-class recall: "
            + ", ".join(f"{k}={v:.4f}" for k, v in calibrated["per_class_recall"].items())
        )
        print(f"  fitted in {elapsed:.1f} s")
        results.append(metrics)
        fitted[model_name] = pipeline

    best = max(results, key=lambda row: row["calibrated"]["macro_f1"])
    best_name = best["model"]
    print(f"\nbest model by calibrated macro-F1: {best_name}")

    best_pipeline = fitted[best_name]
    artifact = best_pipeline.save(RESULTS_DIR / f"model_{best_name}.joblib")

    bundle = {
        "seed": seed,
        "config": config,
        "results": results,
        "best_model": best_name,
        "artifact": str(artifact),
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(bundle, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Global SHAP importance for the best model, for the paper.
    test_idx = best_pipeline.split_indices["test"]
    X_test = df.iloc[test_idx][best_pipeline.features]
    max_rows = shap_config.get("global_max_rows", 2000)
    if len(X_test) > max_rows:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(X_test), size=max_rows, replace=False)
        X_test = X_test.iloc[np.sort(pick)]
    shap_started = time.perf_counter()
    shap_values = best_pipeline.explain(X_test)
    importance = global_importance(shap_values, transformed_feature_names(best_pipeline.pipeline))
    importance.to_csv(RESULTS_DIR / "shap_global_importance.csv", index=False)
    print(
        f"\nglobal SHAP importance over {len(X_test)} rows "
        f"({time.perf_counter() - shap_started:.1f} s):"
    )
    print(importance.head(10).to_string(index=False))

    table = format_metric_table(results)
    report = [
        "# ExoVet - Phase 2 training results",
        "",
        f"Seed {seed}. Grouped split by host star (`kepid`): "
        f"{best['split_sizes']['train']} train / "
        f"{best['split_sizes']['calibration']} calibration / "
        f"{best['split_sizes']['test']} test.",
        "",
        table,
        "",
        "Metrics are computed on the untouched test fold after isotonic",
        "calibration on the held-out calibration fold. `recall@99%` is the",
        "macro average of the per-class recall attainable at 99% precision.",
        "`conformal cov.` is the empirical coverage of the split-conformal",
        f"prediction sets at alpha = {config['conformal'].get('alpha', 0.10)}.",
        "",
        "## Best model",
        "",
        f"`{best_name}` - calibrated macro-F1 "
        f"{best['calibrated']['macro_f1']:.4f}, artifact `{artifact}`.",
        "",
    ]
    (RESULTS_DIR / "train_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nwrote {RESULTS_DIR / 'metrics.json'}")
    print(f"wrote {RESULTS_DIR / 'train_report.md'}")
    print(f"wrote {artifact}")


if __name__ == "__main__":
    main()
