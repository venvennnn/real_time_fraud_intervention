"""Unit and API tests for the fraud intervention service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from fraud.app import create_app
from fraud.engine import InterventionEngine
from fraud.model import FraudModel
from fraud.schemas import Decision, Transaction

BASE_TIME = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def model() -> FraudModel:
    return FraudModel.train()


@pytest.fixture()
def engine(model: FraudModel) -> InterventionEngine:
    return InterventionEngine(model)


def _txn(engine_time_offset_min=0, **overrides):
    defaults = dict(
        transaction_id="t",
        account_id="acct-1",
        amount=25.0,
        merchant_category="grocery",
        country="US",
        latitude=37.7749,
        longitude=-122.4194,
        card_present=True,
        timestamp=BASE_TIME + timedelta(minutes=engine_time_offset_min),
    )
    defaults.update(overrides)
    return Transaction(**defaults)


def test_model_trains_and_separates_classes(model: FraudModel):
    assert model.training_accuracy() > 0.85


def test_legit_transaction_is_approved(engine: InterventionEngine):
    # Warm up account history with normal spending.
    engine.score(_txn(transaction_id="t0", engine_time_offset_min=0))
    result = engine.score(
        _txn(transaction_id="t1", engine_time_offset_min=120, amount=42.0)
    )
    assert result.decision == Decision.APPROVE
    assert result.fraud_probability < 0.35


def test_impossible_travel_is_blocked(engine: InterventionEngine):
    engine.score(
        _txn(
            transaction_id="a",
            engine_time_offset_min=0,
            latitude=37.7749,
            longitude=-122.4194,
        )
    )
    # Two minutes later, a transaction in Lagos: physically impossible.
    result = engine.score(
        _txn(
            transaction_id="b",
            engine_time_offset_min=2,
            amount=800.0,
            country="NG",
            latitude=6.5244,
            longitude=3.3792,
            card_present=False,
        )
    )
    assert result.decision == Decision.BLOCK
    assert any("Impossible travel" in r for r in result.reasons)


def test_high_value_foreign_cnp_is_blocked(engine: InterventionEngine):
    engine.score(_txn(transaction_id="seed", engine_time_offset_min=0))
    result = engine.score(
        _txn(
            transaction_id="big",
            engine_time_offset_min=600,
            amount=3000.0,
            country="GB",
            merchant_category="electronics",
            latitude=51.5074,
            longitude=-0.1278,
            card_present=False,
        )
    )
    assert result.decision == Decision.BLOCK


def test_restricted_merchant_escalates(engine: InterventionEngine):
    engine.score(_txn(transaction_id="seed", engine_time_offset_min=0))
    result = engine.score(
        _txn(
            transaction_id="gc",
            engine_time_offset_min=300,
            amount=50.0,
            merchant_category="gambling",
        )
    )
    assert result.decision in (Decision.CHALLENGE, Decision.BLOCK)


def test_stats_accumulate(engine: InterventionEngine):
    engine.score(_txn(transaction_id="s1", engine_time_offset_min=0))
    engine.score(_txn(transaction_id="s2", engine_time_offset_min=60))
    assert engine.stats.total == 2


def test_api_health_and_score():
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is True

        resp = client.post(
            "/score",
            json={
                "transaction_id": "api-1",
                "account_id": "acct-api",
                "amount": 20.0,
                "merchant_category": "grocery",
                "country": "US",
                "latitude": 37.77,
                "longitude": -122.41,
                "card_present": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] in {"APPROVE", "CHALLENGE", "BLOCK"}
        assert 0.0 <= body["fraud_probability"] <= 1.0
