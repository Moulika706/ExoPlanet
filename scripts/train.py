"""Train and evaluate the ExoVet models (Phase 2).

For every model in the config the script:

1. splits the cleaned KOI table into train / calibration / test folds,
   grouped by host star so sibling KOIs never straddle the split;
2. fits the model on the train fold only;
3. calibrates it on the calibration fold (isotonic by default);
4. fits the split-conformal predictor on the calibrated calibration fold;
5. reports metrics on the untouched test fold.

The best model by calibrated macro-F1 is written to ``results/`` together
with a JSON metric bundle, a Markdown summary, the reliability table of the
best model, and a comparison of the marginal and Mondrian conformal modes.

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

from exovet.data import LABELS, TARGET, clean_labels, load_config, load_raw
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


def conformal_mode_table(
    df: pd.DataFrame, config: Dict[str, Any], model_name: str
) -> List[Dict[str, Any]]:
    """Compare the marginal and Mondrian conformal modes on the same model.

    The marginal mode gives one threshold for all classes and guarantees
    marginal coverage. The Mondrian mode gives one threshold per class and
    guarantees class-conditional coverage, which is the stronger statement
    for a three-class problem with a rare class.
    """
    seed = config.get("seed", 42)
    eval_config = config.get("eval", {})
    splits_config = config.get("splits", {})
    conformal_config = config.get("conformal", {})

    rows: List[Dict[str, Any]] = []
    for mode in ("marginal", "mondrian"):
        pipeline = VettingPipeline(
            model_name=model_name,
            params=config["models"].get(model_name, {}),
            seed=seed,
            calibration_method=config["calibration"]["method"],
            conformal_alpha=conformal_config.get("alpha", 0.10),
            conformal_mode=mode,
            always_include_argmax=conformal_config.get("always_include_argmax", True),
            top_k=config["report"]["top_k"],
        )
        pipeline.fit(
            df,
            n_splits=splits_config.get("n_folds", 5),
            test_fold=eval_config.get("test_fold", 0),
            calibration_fold=eval_config.get("calibration_fold", 1),
        )
        summary = pipeline.metrics["conformal"]
        rows.append(
            {
                "model": model_name,
                "mode": mode,
                "empirical_coverage": summary["empirical_coverage"],
                "per_class_coverage": summary["per_class_coverage"],
                "mean_set_size": summary["mean_set_size"],
                "singleton_rate": summary["singleton_rate"],
                "target_coverage": summary["target_coverage"],
            }
        )
    return rows


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
            n_bins=eval_config.get("ece_bins", 10),
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

    # Reliability table of the best model, before and after calibration.
    # The diagram itself is a figure (Phase 7); the numbers it needs are
    # written here so the plot is reproducible from the results bundle.
    reliability_rows: List[Dict[str, Any]] = []
    for stage, table in best_pipeline.metrics["reliability"].items():
        for row in table:
            reliability_rows.append({"model": best_name, "stage": stage, **row})
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(RESULTS_DIR / "reliability.csv", index=False)

    # Marginal versus Mondrian conformal coverage on the best model.
    print("\n=== conformal modes ===")
    conformal_rows = conformal_mode_table(df, config, best_name)
    for row in conformal_rows:
        print(
            "  {mode}: coverage {cov:.4f}, mean set size {size:.2f}, singleton {single:.4f}".format(
                mode=row["mode"],
                cov=row["empirical_coverage"],
                size=row["mean_set_size"],
                single=row["singleton_rate"],
            )
        )

    bundle = {
        "seed": seed,
        "config": config,
        "results": results,
        "best_model": best_name,
        "artifact": str(artifact),
        "conformal_modes": conformal_rows,
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
        "## Per-class recall",
        "",
        "| model | " + " | ".join(LABELS) + " |",
        "|" + "---|" * (1 + len(LABELS)),
    ]
    for row in results:
        recalls = row["calibrated"]["per_class_recall"]
        cells = " | ".join(f"{recalls.get(cls, float('nan')):.4f}" for cls in LABELS)
        report.append(f"| {row['model']} | {cells} |")

    report += [
        "",
        "## Conformal prediction",
        "",
        f"Target coverage {1 - config['conformal'].get('alpha', 0.10):.2f}, "
        f"mode `{config['conformal'].get('mode', 'marginal')}` by default.",
        "",
        "| model | mode | coverage | "
        + " | ".join(f"cov. {cls}" for cls in LABELS)
        + " | mean set | singleton |",
        "|" + "---|" * (5 + len(LABELS)),
    ]
    for row in conformal_rows:
        per_class = row["per_class_coverage"]
        cells = " | ".join(f"{per_class.get(cls, float('nan')):.4f}" for cls in LABELS)
        report.append(
            "| {model} | {mode} | {cov:.4f} | {cells} | {size:.2f} | {single:.4f} |".format(
                model=row["model"],
                mode=row["mode"],
                cov=row["empirical_coverage"],
                cells=cells,
                size=row["mean_set_size"],
                single=row["singleton_rate"],
            )
        )
    report += [
        "",
        "The marginal mode uses one threshold for all classes and guarantees",
        "marginal coverage. The Mondrian mode uses one threshold per class",
        "and guarantees class-conditional coverage, which is the stronger",
        "statement for the rare CANDIDATE class.",
        "",
        "## Reliability",
        "",
        f"Expected calibration error of the best model: "
        f"{best['calibrated']['expected_calibration_error']:.4f} after calibration "
        f"(was {best['uncalibrated']['expected_calibration_error']:.4f} before).",
        "The full reliability table is in `results/reliability.csv`.",
        "",
        "## Global SHAP importance",
        "",
        f"Mean absolute SHAP value over {len(X_test)} test rows of the best model.",
        "",
        "| rank | feature | mean abs SHAP |",
        "|---|---|---|",
    ]
    for position, row in enumerate(importance.head(10).to_dict("records"), start=1):
        report.append(f"| {position} | {row['feature']} | {row['mean_abs_shap']:.4f} |")
    report += [
        "",
        "How stable this ranking is across folds is measured separately in",
        "`results/shap_stability_report.md`.",
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
