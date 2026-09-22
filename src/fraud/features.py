"""Real-time feature engineering for transaction scoring.

Features are derived from the incoming transaction combined with lightweight
per-account state (velocity, location history, spending baseline) that the
service maintains in memory. This is what makes scoring "real time": each event
is enriched using what we have already seen for that account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schemas import Transaction

FEATURE_ORDER: list[str] = [
    "amount",
    "amount_to_avg_ratio",
    "seconds_since_prev",
    "distance_km",
    "speed_kmh",
    "txn_count_1h",
    "is_foreign",
    "card_not_present",
    "merchant_risk",
]

MERCHANT_RISK: dict[str, float] = {
    "retail": 0.1,
    "grocery": 0.05,
    "travel": 0.4,
    "electronics": 0.5,
    "crypto": 0.9,
    "gambling": 0.85,
    "gift_card": 0.8,
    "cash_advance": 0.7,
}

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in kilometres."""

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def merchant_risk(category: str) -> float:
    return MERCHANT_RISK.get(category.lower().strip(), 0.3)


@dataclass
class AccountState:
    """Rolling state tracked per account for velocity/geo features."""

    home_country: str | None = None
    avg_amount: float = 0.0
    txn_count: int = 0
    last_timestamp: float | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    recent_timestamps: list[float] = field(default_factory=list)


class AccountStateStore:
    """In-memory store of per-account state.

    In production this would be backed by a low-latency store (e.g. Redis);
    the interface is kept intentionally small so it can be swapped out.
    """

    def __init__(self) -> None:
        self._states: dict[str, AccountState] = {}

    def get(self, account_id: str) -> AccountState:
        return self._states.setdefault(account_id, AccountState())

    def reset(self) -> None:
        self._states.clear()


def compute_features(
    txn: Transaction, state: AccountState
) -> dict[str, float]:
    """Derive the model feature vector from a transaction and account state.

    Does not mutate ``state``; call :func:`update_state` after scoring.
    """

    ts = txn.timestamp.timestamp()

    if state.last_timestamp is None:
        seconds_since_prev = 86_400.0  # treat first-seen as a full day gap
    else:
        seconds_since_prev = max(0.0, ts - state.last_timestamp)

    if state.last_lat is None or state.last_lon is None:
        distance_km = 0.0
    else:
        distance_km = haversine_km(
            state.last_lat, state.last_lon, txn.latitude, txn.longitude
        )

    hours = seconds_since_prev / 3600.0
    speed_kmh = distance_km / hours if hours > 1e-6 else distance_km * 3600.0

    window_start = ts - 3600.0
    txn_count_1h = sum(1 for t in state.recent_timestamps if t >= window_start)

    baseline = state.avg_amount if state.avg_amount > 0 else max(txn.amount, 1.0)
    amount_to_avg_ratio = txn.amount / baseline

    home = state.home_country or txn.country
    is_foreign = 0.0 if txn.country == home else 1.0

    return {
        "amount": float(txn.amount),
        "amount_to_avg_ratio": float(amount_to_avg_ratio),
        "seconds_since_prev": float(seconds_since_prev),
        "distance_km": float(distance_km),
        "speed_kmh": float(speed_kmh),
        "txn_count_1h": float(txn_count_1h),
        "is_foreign": is_foreign,
        "card_not_present": 0.0 if txn.card_present else 1.0,
        "merchant_risk": merchant_risk(txn.merchant_category),
    }


def update_state(txn: Transaction, state: AccountState) -> None:
    """Fold a transaction into the account's rolling state."""

    ts = txn.timestamp.timestamp()
    if state.home_country is None:
        state.home_country = txn.country

    if state.txn_count == 0:
        state.avg_amount = txn.amount
    else:
        # Exponential moving average keeps the baseline responsive.
        state.avg_amount = 0.8 * state.avg_amount + 0.2 * txn.amount

    state.txn_count += 1
    state.last_timestamp = ts
    state.last_lat = txn.latitude
    state.last_lon = txn.longitude

    window_start = ts - 3600.0
    state.recent_timestamps = [
        t for t in state.recent_timestamps if t >= window_start
    ]
    state.recent_timestamps.append(ts)


def vectorize(features: dict[str, float]) -> list[float]:
    """Order a feature dict into the fixed model input vector."""

    return [features[name] for name in FEATURE_ORDER]
