"""Fraud scoring model.

A logistic-regression classifier trained at startup on synthetic transaction
features. The synthetic generator encodes the same signals the real-time
feature pipeline produces (velocity, impossible travel, foreign card-not-present
spend, high-risk merchants) so the learned model is consistent with scoring.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_ORDER, vectorize

_RNG_SEED = 42


def _sample_legit(rng: np.random.Generator, n: int) -> np.ndarray:
    amount = rng.gamma(shape=2.0, scale=40.0, size=n)
    amount_ratio = rng.normal(1.0, 0.4, size=n).clip(0.05, 6.0)
    seconds_since_prev = rng.gamma(shape=2.0, scale=30_000.0, size=n)
    distance = rng.gamma(shape=1.2, scale=15.0, size=n)
    hours = np.clip(seconds_since_prev / 3600.0, 1e-3, None)
    speed = distance / hours
    txn_count = rng.poisson(1.0, size=n).astype(float)
    is_foreign = (rng.random(n) < 0.05).astype(float)
    card_not_present = (rng.random(n) < 0.25).astype(float)
    merchant_risk = rng.beta(1.5, 6.0, size=n)
    return np.column_stack(
        [
            amount,
            amount_ratio,
            seconds_since_prev,
            distance,
            speed,
            txn_count,
            is_foreign,
            card_not_present,
            merchant_risk,
        ]
    )


def _sample_fraud(rng: np.random.Generator, n: int) -> np.ndarray:
    amount = rng.gamma(shape=3.0, scale=350.0, size=n)
    amount_ratio = rng.normal(6.0, 3.0, size=n).clip(0.5, 40.0)
    seconds_since_prev = rng.gamma(shape=1.1, scale=400.0, size=n)
    distance = rng.gamma(shape=2.5, scale=800.0, size=n)
    hours = np.clip(seconds_since_prev / 3600.0, 1e-3, None)
    speed = distance / hours
    txn_count = rng.poisson(5.0, size=n).astype(float)
    is_foreign = (rng.random(n) < 0.6).astype(float)
    card_not_present = (rng.random(n) < 0.8).astype(float)
    merchant_risk = rng.beta(5.0, 2.0, size=n)
    return np.column_stack(
        [
            amount,
            amount_ratio,
            seconds_since_prev,
            distance,
            speed,
            txn_count,
            is_foreign,
            card_not_present,
            merchant_risk,
        ]
    )


def build_training_data(
    n_per_class: int = 4000, seed: int = _RNG_SEED
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    legit = _sample_legit(rng, n_per_class)
    fraud = _sample_fraud(rng, n_per_class)
    x = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return x, y


class FraudModel:
    """Wraps a trained scikit-learn pipeline for probability scoring."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self.feature_order = list(FEATURE_ORDER)

    @classmethod
    def train(cls, seed: int = _RNG_SEED) -> "FraudModel":
        x, y = build_training_data(seed=seed)
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
                ),
            ]
        )
        pipeline.fit(x, y)
        return cls(pipeline)

    def score(self, features: dict[str, float]) -> float:
        vector = np.asarray([vectorize(features)], dtype=float)
        proba = float(self._pipeline.predict_proba(vector)[0, 1])
        return proba

    def training_accuracy(self, seed: int = _RNG_SEED) -> float:
        x, y = build_training_data(seed=seed + 1)
        return float(self._pipeline.score(x, y))
