"""End-to-end vetting pipeline for ExoVet.

Ties together the pieces of the pipeline into one object:

    features -> model -> calibration -> conformal prediction set -> SHAP -> report

The object is what the API will serve (Phase 6) and what the training
script stores on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .calibration import apply_calibrator, fit_calibrator
from .conformal import SplitConformal
from .data import LABELS
from .eval import evaluate
from .explain import compute_shap_values, local_explanation_frame, transformed_feature_names
from .models import build_model
from .report import build_report, check_report_features
from .splits import assert_disjoint_groups, train_calibration_test


@dataclass
class VettingResult:
    """One vetted object."""

    predicted_class: str
    probability: float
    probabilities: Dict[str, float]
    prediction_set: List[str]
    needs_review: bool
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    report: str = ""
    raw_probabilities: Dict[str, float] = field(default_factory=dict)


class VettingPipeline:
    """Model + calibrator + conformal predictor + explanation."""

    def __init__(
        self,
        model_name: str = "xgboost",
        features: Optional[Sequence[str]] = None,
        include_leakage: bool = False,
        params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        calibration_method: str = "isotonic",
        conformal_alpha: float = 0.10,
        conformal_mode: str = "marginal",
        always_include_argmax: bool = True,
        shap_perturbation: str = "tree_path",
        classes: Sequence[str] = LABELS,
        top_k: int = 5,
    ) -> None:
        self.model_name = model_name
        self.include_leakage = include_leakage
        self.features = list(features) if features is not None else None
        self.params = dict(params or {})
        self.seed = seed
        self.calibration_method = calibration_method
        self.conformal_alpha = conformal_alpha
        self.conformal_mode = conformal_mode
        self.always_include_argmax = always_include_argmax
        self.shap_perturbation = shap_perturbation
        self.classes = tuple(classes)
        self.top_k = top_k

        self.pipeline = None
        self.label_encoder = None
        self.calibrator = None
        self.conformal: Optional[SplitConformal] = None
        self.shap_values: Optional[np.ndarray] = None
        self.shap_names: List[str] = []
        self.background_: Optional[np.ndarray] = None
        self.metrics: Dict[str, Any] = {}
        self.split_indices: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_fold: int = 0,
        calibration_fold: int = 1,
        group_column: str = "kepid",
        target_column: str = "koi_disposition",
        split_strategy: str = "grouped",
    ) -> "VettingPipeline":
        """Fit model, calibrator and conformal predictor on grouped folds.

        *split_strategy* is ``grouped`` (the default, host stars never
        straddle the split) or ``random`` (the leaky baseline used by the
        split ablation).
        """
        from .features import feature_list
        from .splits import random_train_calibration_test

        features = self.features or feature_list(self.include_leakage)
        self.features = features

        if split_strategy == "grouped":
            train_idx, cal_idx, test_idx = train_calibration_test(
                df,
                n_splits=n_splits,
                test_fold=test_fold,
                calibration_fold=calibration_fold,
                group_column=group_column,
                target_column=target_column,
                seed=self.seed,
            )
        elif split_strategy == "random":
            train_idx, cal_idx, test_idx = random_train_calibration_test(
                df,
                n_splits=n_splits,
                test_fold=test_fold,
                calibration_fold=calibration_fold,
                target_column=target_column,
                seed=self.seed,
            )
        elif split_strategy == "sibling_leak":
            # Same test rows as the grouped split, but the training pool is
            # drawn at random, so host stars in the test fold also have
            # their sibling KOIs in training.
            from .splits import sibling_leak_train_calibration_test

            _, _, grouped_test = train_calibration_test(
                df,
                n_splits=n_splits,
                test_fold=test_fold,
                calibration_fold=calibration_fold,
                group_column=group_column,
                target_column=target_column,
                seed=self.seed,
            )
            train_idx, cal_idx, test_idx = sibling_leak_train_calibration_test(
                df, grouped_test, seed=self.seed
            )
        else:
            raise ValueError(f"unknown split strategy {split_strategy!r}")

        if split_strategy == "grouped":
            assert_disjoint_groups(df, train_idx, test_idx, group_column)
            assert_disjoint_groups(df, train_idx, cal_idx, group_column)
            assert_disjoint_groups(df, cal_idx, test_idx, group_column)
        self.split_indices = {
            "train": [int(i) for i in train_idx],
            "calibration": [int(i) for i in cal_idx],
            "test": [int(i) for i in test_idx],
        }

        X_train = df.iloc[train_idx][features]
        y_train = df.iloc[train_idx][target_column].to_numpy()
        X_cal = df.iloc[cal_idx][features]
        y_cal = df.iloc[cal_idx][target_column].to_numpy()
        X_test = df.iloc[test_idx][features]
        y_test = df.iloc[test_idx][target_column].to_numpy()

        # The label encoder lives inside the model pipeline (XGBoost rejects
        # string labels). Its alphabetical order (CANDIDATE, CONFIRMED,
        # FALSE POSITIVE) differs from the canonical label order used
        # everywhere else, so the mapping is explicit.
        self.pipeline = build_model(
            self.model_name,
            params=self.params,
            seed=self.seed,
            feature_names=features,
        )
        self.pipeline.fit(X_train, y_train)
        self.label_encoder = self.pipeline.named_steps["classifier"].encoder_

        # A fixed reference sample of the training fold, used as the SHAP
        # background so that explanations do not depend on the row being
        # explained.
        rng = np.random.default_rng(self.seed)
        background_rows = min(100, len(X_train))
        pick = rng.choice(len(X_train), size=background_rows, replace=False)
        self.background_ = self.pipeline.named_steps["preprocessor"].transform(X_train.iloc[pick])

        raw_cal = self._reorder(self.pipeline.predict_proba(X_cal))
        self.calibrator = fit_calibrator(self.calibration_method, raw_cal, y_cal, self.classes)
        cal_proba = apply_calibrator(self.calibrator, raw_cal)

        self.conformal = SplitConformal(
            alpha=self.conformal_alpha,
            mode=self.conformal_mode,
            classes=self.classes,
            always_include_argmax=self.always_include_argmax,
        ).fit(cal_proba, y_cal)

        raw_test = self._reorder(self.pipeline.predict_proba(X_test))
        test_proba = apply_calibrator(self.calibrator, raw_test)
        self.metrics = {
            "model": self.model_name,
            "include_leakage": self.include_leakage,
            "split_strategy": split_strategy,
            "n_features": len(features),
            "features": features,
            "uncalibrated": evaluate(y_test, raw_test, self.classes),
            "calibrated": evaluate(y_test, test_proba, self.classes),
            "conformal": self.conformal.summary(test_proba, y_test),
            "split_sizes": {
                "train": len(train_idx),
                "calibration": len(cal_idx),
                "test": len(test_idx),
            },
        }
        return self

    # ------------------------------------------------------------------
    # explanation
    # ------------------------------------------------------------------
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """Compute and cache SHAP values for *X*."""
        if self.pipeline is None:
            raise RuntimeError("pipeline is not fitted")
        self.shap_values = compute_shap_values(
            self.pipeline,
            X,
            self.classes,
            background=self.background_,
            seed=self.seed,
            perturbation=self.shap_perturbation,
        )
        self.shap_names = transformed_feature_names(self.pipeline)
        return self.shap_values

    def explain_frame(self, X: pd.DataFrame, row_position: int = 0) -> pd.DataFrame:
        """Ranked SHAP frame for one row of *X*."""
        if self.shap_values is None:
            self.explain(X)
        class_index = int(np.argmax(self._calibrated_proba(X)[row_position]))
        return local_explanation_frame(
            self.shap_values,
            X,
            self.shap_names,
            class_index,
            row_position=row_position,
            top_k=self.top_k,
        )

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def _reorder(self, proba: np.ndarray) -> np.ndarray:
        """Reorder probability columns into the canonical class order."""
        if self.label_encoder is None:
            return np.asarray(proba, dtype=float)
        encoder_classes = list(self.label_encoder.classes_)
        order = [encoder_classes.index(cls) for cls in self.classes]
        return np.asarray(proba, dtype=float)[:, order]

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("pipeline is not fitted")
        return self._reorder(self.pipeline.predict_proba(X[self.features]))

    def _calibrated_proba(self, X: pd.DataFrame) -> np.ndarray:
        return apply_calibrator(self.calibrator, self._raw_proba(X))

    def predict_frame(self, X: pd.DataFrame) -> List[VettingResult]:
        """Vet every row of *X* and return one result per row."""
        raw = self._raw_proba(X)
        proba = apply_calibrator(self.calibrator, raw)
        sets = self.conformal.predict_sets(proba) if self.conformal is not None else None
        self.explain(X)

        results: List[VettingResult] = []
        for position in range(len(X)):
            class_index = int(np.argmax(proba[position]))
            predicted = self.classes[class_index]
            prediction_set = sets[position] if sets is not None else None
            frame = local_explanation_frame(
                self.shap_values,
                X,
                self.shap_names,
                class_index,
                row_position=position,
                top_k=self.top_k,
            )
            report = build_report(
                frame,
                proba[position],
                self.classes,
                prediction_set=prediction_set,
                top_k=self.top_k,
            )
            violations = check_report_features(
                report["text"], self.features, [r["feature"] for r in report["top_features"]]
            )
            results.append(
                VettingResult(
                    predicted_class=predicted,
                    probability=float(proba[position, class_index]),
                    probabilities={
                        cls: float(proba[position, i]) for i, cls in enumerate(self.classes)
                    },
                    raw_probabilities={
                        cls: float(raw[position, i]) for i, cls in enumerate(self.classes)
                    },
                    prediction_set=list(prediction_set) if prediction_set else [predicted],
                    needs_review=bool(report["needs_review"]),
                    top_features=report["top_features"],
                    report=report["text"],
                )
            )
            if violations:
                raise AssertionError(
                    f"report mentioned features outside the top-{self.top_k}: {violations}"
                )
        return results

    def predict_one(self, values: Dict[str, float]) -> VettingResult:
        """Vet a single object given a mapping of feature name to value."""
        row = {feature: values.get(feature, np.nan) for feature in self.features}
        frame = pd.DataFrame([row])
        return self.predict_frame(frame)[0]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Serialise the fitted pipeline with joblib."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "features": self.features,
                "include_leakage": self.include_leakage,
                "params": self.params,
                "seed": self.seed,
                "calibration_method": self.calibration_method,
                "conformal_alpha": self.conformal_alpha,
                "conformal_mode": self.conformal_mode,
                "always_include_argmax": self.always_include_argmax,
                "shap_perturbation": self.shap_perturbation,
                "classes": self.classes,
                "top_k": self.top_k,
                "pipeline": self.pipeline,
                "label_encoder": self.label_encoder,
                "background": self.background_,
                "calibrator": self.calibrator,
                "conformal": self.conformal,
                "metrics": self.metrics,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VettingPipeline":
        """Load a pipeline saved by :meth:`save`."""
        import joblib

        state = joblib.load(Path(path))
        instance = cls(
            model_name=state["model_name"],
            features=state["features"],
            include_leakage=state["include_leakage"],
            params=state["params"],
            seed=state["seed"],
            calibration_method=state["calibration_method"],
            conformal_alpha=state["conformal_alpha"],
            conformal_mode=state["conformal_mode"],
            always_include_argmax=state.get("always_include_argmax", True),
            shap_perturbation=state.get("shap_perturbation", "tree_path"),
            classes=state["classes"],
            top_k=state["top_k"],
        )
        instance.pipeline = state["pipeline"]
        instance.label_encoder = state.get("label_encoder")
        instance.background_ = state.get("background")
        instance.calibrator = state["calibrator"]
        instance.conformal = state["conformal"]
        instance.metrics = state.get("metrics", {})
        return instance
