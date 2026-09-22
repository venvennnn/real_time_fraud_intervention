"""Pydantic schemas for the fraud intervention API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """Intervention decision returned for a scored transaction."""

    APPROVE = "APPROVE"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"


class Transaction(BaseModel):
    """A single payment authorization request to be scored in real time."""

    transaction_id: str = Field(..., description="Unique transaction identifier.")
    account_id: str = Field(..., description="Account/card holder identifier.")
    amount: float = Field(..., ge=0, description="Transaction amount in USD.")
    merchant_category: str = Field(
        "retail", description="Merchant category, e.g. retail, travel, gambling."
    )
    country: str = Field("US", description="ISO-3166 alpha-2 country of the merchant.")
    latitude: float = Field(0.0, ge=-90.0, le=90.0, description="Merchant latitude.")
    longitude: float = Field(
        0.0, ge=-180.0, le=180.0, description="Merchant longitude."
    )
    card_present: bool = Field(
        True, description="Whether the physical card was present (chip/tap)."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the transaction occurred (UTC).",
    )


class InterventionResult(BaseModel):
    """Result of scoring a transaction and applying the intervention policy."""

    transaction_id: str
    account_id: str
    decision: Decision
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)


class DecisionStats(BaseModel):
    """Aggregate counts of decisions made since startup."""

    total: int = 0
    approve: int = 0
    challenge: int = 0
    block: int = 0


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    version: str
