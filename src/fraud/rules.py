"""Deterministic business rules layered on top of the ML score.

Rules encode hard risk policies that must hold regardless of the model, and
they can escalate (but never de-escalate) the model-driven decision. Each rule
returns a minimum decision severity plus a human-readable reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import Decision, Transaction

# High-value, impossible-travel and velocity thresholds.
IMPOSSIBLE_TRAVEL_KMH = 900.0
HIGH_VALUE_USD = 2_000.0
VELOCITY_COUNT_1H = 6
BLOCKED_MERCHANTS = {"gambling", "crypto", "gift_card"}

_SEVERITY = {Decision.APPROVE: 0, Decision.CHALLENGE: 1, Decision.BLOCK: 2}


@dataclass
class RuleOutcome:
    min_decision: Decision
    reasons: list[str]


def apply_rules(txn: Transaction, features: dict[str, float]) -> RuleOutcome:
    reasons: list[str] = []
    min_decision = Decision.APPROVE

    def escalate(to: Decision, reason: str) -> None:
        nonlocal min_decision
        reasons.append(reason)
        if _SEVERITY[to] > _SEVERITY[min_decision]:
            min_decision = to

    if features["speed_kmh"] >= IMPOSSIBLE_TRAVEL_KMH:
        escalate(
            Decision.BLOCK,
            f"Impossible travel: {features['speed_kmh']:.0f} km/h since last txn",
        )

    if features["txn_count_1h"] >= VELOCITY_COUNT_1H:
        escalate(
            Decision.CHALLENGE,
            f"High velocity: {int(features['txn_count_1h'])} txns in last hour",
        )

    if (
        txn.amount >= HIGH_VALUE_USD
        and features["card_not_present"] == 1.0
        and features["is_foreign"] == 1.0
    ):
        escalate(
            Decision.BLOCK,
            "High-value foreign card-not-present transaction",
        )

    if txn.merchant_category.lower().strip() in BLOCKED_MERCHANTS:
        escalate(
            Decision.CHALLENGE,
            f"Restricted merchant category: {txn.merchant_category}",
        )

    return RuleOutcome(min_decision=min_decision, reasons=reasons)


def max_decision(a: Decision, b: Decision) -> Decision:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b
