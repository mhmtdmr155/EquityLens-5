"""EquityLens-5 Machine Learning Experiment Metrics Route."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas import ExperimentMetricsResponse, ModelScoreDetail

logger = logging.getLogger("EquityLens5.ExperimentsAPI")
router = APIRouter(prefix="/experiments", tags=["Experiments"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
M2_METRICS_PATH = (
    PROJECT_ROOT / "models" / "experiments" / "market_m2" / "m2_experiment_metrics.json"
)


def _extract_score_detail(raw: dict[str, float | int | None]) -> ModelScoreDetail:
    """Helper to convert raw metric mapping into ModelScoreDetail."""
    roc_auc_val = round(float(raw["roc_auc"]), 4) if raw.get("roc_auc") is not None else None
    return ModelScoreDetail(
        accuracy=round(float(raw.get("accuracy", 0.0)), 4),
        balanced_accuracy=round(float(raw.get("balanced_accuracy", 0.0)), 4),
        precision=round(float(raw.get("precision", 0.0)), 4),
        recall=round(float(raw.get("recall", 0.0)), 4),
        f1_score=round(float(raw.get("f1_score", 0.0)), 4),
        roc_auc=roc_auc_val,
    )


@router.get(
    "/metrics",
    response_model=ExperimentMetricsResponse,
    summary="Get M0, M1, M2 and Baseline machine learning evaluation metrics",
    responses={
        200: {"description": "Experiment metrics retrieved successfully"},
        503: {"description": "Experiment metrics file not available"},
    },
)
def get_experiment_metrics() -> ExperimentMetricsResponse:
    """Retrieve saved offline experiment evaluation metrics for model comparisons.

    Covers retrospective test split results and expanding window cross-validation
    averages across B0a, B0b, M0, M1, and M2 experiments.
    """
    if not M2_METRICS_PATH.exists():
        logger.warning("Experiment metrics file not found at %s", M2_METRICS_PATH)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Experiment metrics artifact not found on disk.",
        )

    try:
        with open(M2_METRICS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.exception("Failed reading experiment metrics: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed loading experiment metrics: {exc}",
        ) from exc

    # Parse retrospective ablation models
    retrospective_models: dict[str, ModelScoreDetail] = {}
    retro_data = data.get("retrospective_ablation", {}).get("models", {})
    for key, val in retro_data.items():
        if isinstance(val, dict):
            retrospective_models[key] = _extract_score_detail(val)

    # Parse expanding window CV averages
    cv_averages: dict[str, ModelScoreDetail] = {}
    cv_data = data.get("expanding_window_cv", {}).get("fold_averages", {})
    for key, val in cv_data.items():
        if isinstance(val, dict):
            cv_averages[key] = _extract_score_detail(val)

    as_of_date = (
        data.get("retrospective_ablation", {})
        .get("boundaries", {})
        .get("max_available_date", "2026-10-30")
    )

    disclaimer = (
        "Offline research metrics from chronologically purged test and cross-validation splits. "
        "Algorithmic scores do not constitute live financial signals or guaranteed returns."
    )

    return ExperimentMetricsResponse(
        status="SUCCESS",
        as_of_date=as_of_date,
        retrospective_models=retrospective_models,
        expanding_cv_averages=cv_averages,
        disclaimer=disclaimer,
    )
