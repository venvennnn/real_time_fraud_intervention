"""Intervention engine: combine model score, rules, and account state."""

from __future__ import annotations

from .features import (
    AccountStateStore,
    compute_features,
    update_state,
)
from .model import FraudModel
from .rules import apply_rules, max_decision
from .schemas import Decision, DecisionStats, InterventionResult, Transaction

# Probability thresholds for the model-driven decision.
CHALLENGE_THRESHOLD = 0.35
BLOCK_THRESHOLD = 0.75


class InterventionEngine:
    """Scores transactions and applies the real-time intervention policy."""

    def __init__(self, model: FraudModel, store: AccountStateStore | None = None):
        self._model = model
        self._store = store or AccountStateStore()
        self._stats = DecisionStats()

    @property
    def stats(self) -> DecisionStats:
        return self._stats

    def _model_decision(self, probability: float) -> Decision:
        if probability >= BLOCK_THRESHOLD:
            return Decision.BLOCK
        if probability >= CHALLENGE_THRESHOLD:
            return Decision.CHALLENGE
        return Decision.APPROVE

    def score(self, txn: Transaction) -> InterventionResult:
        state = self._store.get(txn.account_id)
        features = compute_features(txn, state)

        probability = self._model.score(features)
        decision = self._model_decision(probability)

        reasons: list[str] = []
        if decision != Decision.APPROVE:
            reasons.append(f"Model fraud probability {probability:.2f}")

        rule_outcome = apply_rules(txn, features)
        decision = max_decision(decision, rule_outcome.min_decision)
        reasons.extend(rule_outcome.reasons)

        if not reasons:
            reasons.append("No risk signals detected")

        # State is updated after scoring so features reflect prior history only.
        update_state(txn, state)
        self._record(decision)

        return InterventionResult(
            transaction_id=txn.transaction_id,
            account_id=txn.account_id,
            decision=decision,
            fraud_probability=round(probability, 4),
            reasons=reasons,
            features={k: round(v, 4) for k, v in features.items()},
        )

    def _record(self, decision: Decision) -> None:
        self._stats.total += 1
        if decision == Decision.APPROVE:
            self._stats.approve += 1
        elif decision == Decision.CHALLENGE:
            self._stats.challenge += 1
        else:
            self._stats.block += 1

    def reset(self) -> None:
        self._store.reset()
        self._stats = DecisionStats()
