"""Replay a realistic transaction stream against the running service.

Simulates a normal spending pattern for an account followed by an account
takeover attack (rapid, high-value, foreign, card-not-present transactions and
impossible travel) and prints the intervention decision for each event.

Usage:
    python scripts/demo_stream.py [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timedelta, timezone

import httpx

# (lat, lon) for a few cities used to build a geographically coherent stream.
SF = (37.7749, -122.4194)
LA = (34.0522, -118.2437)
LONDON = (51.5074, -0.1278)
LAGOS = (6.5244, 3.3792)


def build_stream() -> list[dict]:
    base = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    # Use a fresh account id each run so per-account state (which the service
    # keeps in memory across requests) always starts clean and the stream is
    # reproducible regardless of prior traffic.
    acct = f"acct-{uuid.uuid4().hex[:8]}"
    events: list[dict] = []

    def txn(txn_id, minutes, amount, category, country, loc, card_present):
        lat, lon = loc
        return {
            "transaction_id": txn_id,
            "account_id": acct,
            "amount": amount,
            "merchant_category": category,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "card_present": card_present,
            "timestamp": (base + timedelta(minutes=minutes)).isoformat(),
        }

    # --- Normal, in-region spending (should APPROVE) ---
    events.append(txn("t1", 0, 12.50, "grocery", "US", SF, True))
    events.append(txn("t2", 90, 46.00, "retail", "US", SF, True))
    events.append(txn("t3", 240, 88.00, "travel", "US", LA, True))

    # --- Account takeover: impossible travel + high-value foreign CNP burst ---
    events.append(txn("t4", 250, 1500.00, "electronics", "GB", LONDON, False))
    events.append(txn("t5", 252, 2400.00, "gift_card", "NG", LAGOS, False))
    events.append(txn("t6", 253, 3100.00, "crypto", "NG", LAGOS, False))
    events.append(txn("t7", 254, 2750.00, "electronics", "NG", LAGOS, False))
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    stream = build_stream()
    print(f"Streaming {len(stream)} transactions to {args.url}\n")
    print(f"{'TXN':<5}{'AMOUNT':>10} {'MCC':<12}{'CTRY':<6}{'DECISION':<10}{'P(fraud)':>9}  REASONS")
    print("-" * 100)

    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        health = client.get("/health").json()
        print(f"[health] {health}\n")
        for event in stream:
            resp = client.post("/score", json=event)
            resp.raise_for_status()
            result = resp.json()
            reasons = "; ".join(result["reasons"])
            print(
                f"{event['transaction_id']:<5}"
                f"{event['amount']:>10.2f} "
                f"{event['merchant_category']:<12}"
                f"{event['country']:<6}"
                f"{result['decision']:<10}"
                f"{result['fraud_probability']:>9.2f}  {reasons}"
            )

        stats = client.get("/stats").json()
        print("\n[stats]", stats)


if __name__ == "__main__":
    main()
