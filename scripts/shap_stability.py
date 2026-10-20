"""SHAP feature-stability analysis across folds (Phase 2).

The training report shows one global SHAP table, computed on one test fold.
This script asks the harder question: does that ranking survive a different
split? The model is refitted on every grouped fold in turn, the global SHAP
importance is computed on the fold that was held out, and the rankings are
compared with each other.

If the top features change from fold to fold, then the explanation shown to
a user is an artefact of one split and the report generator should say so.
If they do not, the explanation is a property of the model.

Usage:
    python scripts/shap_stability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exovet.data import clean_labels, load_config, load_raw
from exovet.stability import cross_fold_analysis, ranking_stability, stability_markdown

RESULTS_DIR = Path("results")


def main() -> None:
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stability_config = config.get("stability", {})
    model_name = stability_config.get("model", "xgboost")
    top_k = stability_config.get("top_k", config.get("report", {}).get("top_k", 5))
    seed = config.get("seed", 42)
    n_splits = config.get("splits", {}).get("n_folds", 5)

    df = clean_labels(load_raw(config["data"]["raw_csv"]))
    print(f"data: {len(df)} rows")
    print(f"model: {model_name}, folds: {n_splits}, seed: {seed}")

    analysis = cross_fold_analysis(
        df,
        model_name=model_name,
        params=config["models"].get(model_name, {}),
        n_splits=n_splits,
        seed=seed,
        max_rows=stability_config.get("max_rows", 1500),
    )
    stability = ranking_stability(analysis["importance"], top_k=top_k)

    for row in analysis["scores"]:
        print(f"  fold {row['fold']}: macro-F1 {row['macro_f1']:.4f} on {row['n_test']} rows")
    print(
        f"top-{top_k} overlap across folds: {stability['top_k_overlap_mean']:.4f} "
        f"(worst pair {stability['top_k_overlap_min']:.4f})"
    )
    print(f"mean Kendall tau between folds: {stability['mean_kendall_tau']:.4f}")

    importance = analysis["importance"]
    importance.to_csv(RESULTS_DIR / "shap_fold_importance.csv", index=False)

    bundle: Dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "n_splits": n_splits,
        "top_k": top_k,
        "per_fold_scores": analysis["scores"],
        "stability": {
            key: value for key, value in stability.items() if key not in ("consensus_ranking",)
        },
        "consensus_ranking": stability["consensus_ranking"],
    }
    (RESULTS_DIR / "shap_stability.json").write_text(
        json.dumps(bundle, indent=2, default=str) + "\n", encoding="utf-8"
    )

    report = stability_markdown(analysis, stability, top_k=top_k)
    (RESULTS_DIR / "shap_stability_report.md").write_text(report, encoding="utf-8")

    print(f"\nwrote {RESULTS_DIR / 'shap_stability.json'}")
    print(f"wrote {RESULTS_DIR / 'shap_stability_report.md'}")
    print(f"wrote {RESULTS_DIR / 'shap_fold_importance.csv'}")


if __name__ == "__main__":
    main()
