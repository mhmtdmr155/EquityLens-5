"""FastAPI route for 5-day stock price direction machine learning predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.features import FEATURE_COLUMNS, compute_features_for_series
from backend.app.models import Company, PriceBar, Security
from backend.app.schemas import PredictionResponse

logger = logging.getLogger("EquityLens5.PredictionsAPI")
router = APIRouter(prefix="/predictions", tags=["Predictions"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "direction_5d_model.joblib"


def get_loaded_model(model_path: Path | None = None) -> Any:
    """Load serialized model artifact from disk or raise 503 if missing."""
    target_path = model_path if model_path is not None else MODEL_PATH
    if not target_path.exists():
        logger.warning("Prediction request rejected: model artifact not found at %s", target_path)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact not found. Please run baseline model training first.",
        )
    try:
        return joblib.load(target_path)
    except Exception as exc:
        logger.exception("Failed loading model artifact: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed loading model artifact: {exc}",
        ) from exc


@router.get(
    "/{ticker}",
    response_model=PredictionResponse,
    summary="Get 5-day price direction machine learning prediction",
    responses={
        200: {"description": "Prediction computed successfully"},
        404: {"description": "Ticker not found in database"},
        422: {"description": "Insufficient price history for feature warmup"},
        503: {"description": "Model artifact unavailable or failed to load"},
    },
)
def get_stock_direction_prediction(
    ticker: str,
    db: Annotated[Session, Depends(get_db)],
) -> PredictionResponse:
    """Compute algorithmic 5-trading-day price direction prediction for a tracked ticker.

    Features are derived from historical end-of-day price bars in the database.
    Does not constitute live market financial advice.
    """
    clean_ticker = ticker.strip().upper()

    # 1. Resolve security and company
    stmt_sec = select(Security).where(Security.ticker == clean_ticker)
    security = db.execute(stmt_sec).scalar_one_or_none()
    if security is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{clean_ticker}' not found.",
        )

    stmt_comp = select(Company).where(Company.id == security.company_id)
    company = db.execute(stmt_comp).scalar_one()

    # 2. Retrieve chronological price history
    stmt_bars = (
        select(PriceBar)
        .where(PriceBar.security_id == security.id)
        .order_by(PriceBar.session_date.asc())
    )
    bars = db.execute(stmt_bars).scalars().all()

    # Warmup check: MACD and 20-period moving averages require at least 34 bars
    if len(bars) < 34:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Insufficient price history for ticker '{clean_ticker}': "
                f"found {len(bars)} bars, requires at least 34 bars for feature warmup."
            ),
        )

    # 3. Construct DataFrame and extract features
    data = [
        {
            "session_date": str(b.session_date),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": int(b.volume),
        }
        for b in bars
    ]
    df_bars = pd.DataFrame(data)
    feat_df = compute_features_for_series(df_bars)

    # 4. Extract most recent feature vector
    latest_row = feat_df.iloc[-1]
    as_of_date = str(latest_row["session_date"])

    # Verify no NaN values in features
    if latest_row[FEATURE_COLUMNS].isna().any():
        nan_cols = latest_row[FEATURE_COLUMNS][latest_row[FEATURE_COLUMNS].isna()].index.tolist()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Incomplete features for ticker '{clean_ticker}' as of {as_of_date}: {nan_cols}"
            ),
        )

    X_latest = pd.DataFrame([latest_row[FEATURE_COLUMNS]])

    # 5. Load model and predict
    model = get_loaded_model()
    pred_dir = int(model.predict(X_latest)[0])

    prob_up = 0.5
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_latest)
            if probs.shape[1] >= 2:
                prob_up = round(float(probs[0, 1]), 4)
            else:
                prob_up = round(float(probs[0, 0]), 4)
        except Exception:
            prob_up = 0.5

    # 6. Format model metadata and warning
    model_name = getattr(model, "__class__", type(model)).__name__
    warning_text = (
        f"Notice: Prediction is derived from historical end-of-day data as of {as_of_date} "
        "(Alpaca IEX raw). It is NOT live real-time market data or financial advice. "
        "probability_up is an uncalibrated algorithmic score, not a guaranteed return probability."
    )

    return PredictionResponse(
        ticker=clean_ticker,
        company_name=company.company_name,
        model_name=model_name,
        model_version="1.0.0",
        horizon_trading_days=5,
        as_of_date=as_of_date,
        predicted_direction=pred_dir,
        probability_up=prob_up,
        data_source="Alpaca Market Data v2 (IEX raw)",
        warning=warning_text,
    )
