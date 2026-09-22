# real_time_fraud_intervention

A real-time payment **fraud intervention** service. It scores incoming
transactions as they arrive and decides whether to **approve**, **challenge**
(step-up authentication), or **block** them, combining a machine-learning risk
score with deterministic business rules and per-account velocity/geo state.

## How it works

```
transaction ──▶ feature engineering ──▶ ML score ──▶ intervention policy ──▶ decision
                 (per-account state)    (logistic     (+ hard rules)          APPROVE /
                                         regression)                          CHALLENGE /
                                                                              BLOCK
```

- **Feature engineering** (`src/fraud/features.py`): derives real-time signals
  from the transaction plus rolling per-account state — spend-vs-baseline,
  velocity (transactions/hour), impossible-travel speed (haversine distance
  over elapsed time), foreign-country and card-not-present flags, merchant risk.
- **Model** (`src/fraud/model.py`): a scikit-learn logistic-regression pipeline
  trained at startup on synthetic data that mirrors those signals.
- **Rules** (`src/fraud/rules.py`): hard policies (impossible travel, high-value
  foreign card-not-present spend, velocity bursts, restricted merchants) that
  can escalate — but never soften — the model decision.
- **Engine** (`src/fraud/engine.py`): ties it together and tracks decision stats.
- **API** (`src/fraud/app.py`): FastAPI service exposing `/health`, `/score`,
  and `/stats`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the service
uvicorn fraud.app:app --host 0.0.0.0 --port 8000

# In another shell, replay a demo transaction stream
python scripts/demo_stream.py
```

## API

| Method | Path      | Description                                   |
| ------ | --------- | --------------------------------------------- |
| GET    | `/health` | Liveness and model-readiness check.           |
| POST   | `/score`  | Score one transaction, return an intervention.|
| GET    | `/stats`  | Aggregate decision counts since startup.      |

Example:

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "transaction_id": "t1", "account_id": "acct-1", "amount": 3200,
  "merchant_category": "electronics", "country": "NG",
  "latitude": 6.52, "longitude": 3.38, "card_present": false
}' | jq
```

## Tests

```bash
pytest
```

## Cloud Agent environment

This repository ships a Cloud Agent environment (`.cursor/environment.json`).
On the default base image its `install` step (`.cursor/install.sh`) installs the
package with pip, and a `server` terminal launches the API with
`python3 -m uvicorn fraud.app:app` on port 8000.
