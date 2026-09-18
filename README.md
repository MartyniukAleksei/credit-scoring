# Credit Scoring System

Production-style ML system that predicts the probability of serious loan delinquency
(90+ days past due within 2 years) from a borrower's credit profile, and explains
*why* the model flagged a given applicant as risky.

Built as part of a self-directed AI/ML Engineer training roadmap — not a course
project, but an end-to-end system with a reproducible pipeline, a served API,
and versioned data/models.

## Why this matters

A credit scoring model makes two kinds of mistakes, and they are not equally
expensive:

- **False negative** — the model says "low risk," but the borrower defaults.
  The lender loses the unpaid principal plus collection costs. This is usually
  the costlier error per incident.
- **False positive** — the model says "high risk," but the borrower would have
  repaid. The lender loses a creditworthy customer and future interest revenue,
  and — at scale — risks systematically penalizing applicants who didn't deserve
  the flag.

Because the real cost of each error depends on the lender's own risk appetite and
margins, this system does **not** hardcode an approve/deny decision. It returns a
calibrated probability; where to draw the accept/reject line is a business
decision for risk/finance stakeholders, not a modeling one.

The target class is also imbalanced — roughly 6–7% positive — which is why the
model is tuned and evaluated on **PR-AUC** rather than accuracy: at this base
rate, a model that never flags anyone would already be ~93% "accurate" and
completely useless.

## What it does

Two inference modes over the same trained pipeline:

- **`POST /predict`** — real-time scoring for a single application. Returns a
  probability and the top SHAP-attributed risk factors for that specific
  applicant, so a human reviewer sees *why*, not just a number.
- **Batch scoring** — scores an entire file/portfolio of applications at once
  through the same registered pipeline, for nightly/offline runs rather than one
  request at a time.

## Architecture

```
raw CSV → ingest → validate (Pandera) → train/test split
        → tune (Optuna, 5-fold CV) → train (LightGBM) → evaluate
        → register (only overwrites the shipped model if it's not worse)
```

- **Data contract** — a [Pandera](https://pandera.readthedocs.io) schema
  validates every batch of data before it reaches the pipeline. Structural
  violations (negative values, invalid target, unexpected nulls) fail the run
  loudly; statistically unusual-but-plausible values (sentinel codes,
  historical-norm outliers) only warn. The same "schema as contract" principle
  is applied one level down by Pydantic at the API boundary — a single
  request instead of a whole table.
- **Feature engineering lives inside `sklearn.Pipeline`**, fit together with the
  model as one atomic object — the standard guard against train/serve skew: the
  preprocessing that ran at training time is *guaranteed* to be the same
  preprocessing that runs at inference time, because it's literally the same
  fitted object.
- **Model** — LightGBM, hyperparameters tuned via Optuna (30 trials, 5-fold
  stratified CV, optimizing PR-AUC), benchmarked against a logistic-regression
  baseline with WOE-encoded features.
- **Orchestration** — [Prefect](https://www.prefect.io). Both training and
  batch scoring are DAGs of `@task`s under a `@flow`, giving per-step run
  history and isolated failure visibility instead of one opaque script.
- **Data & model versioning** — [DVC](https://dvc.org). The raw dataset and the
  serialized model are content-addressed and stored outside git; git tracks
  only small `.dvc` pointer files. A given commit hash reproduces the exact
  dataset and model that produced it.
- **Retraining is human-gated by design** — `register()` compares the new
  model's PR-AUC against the currently shipped one (tracked in
  `models/metrics.json`) and only overwrites it if the new model is at least as
  good. Retraining is expensive and changes what serves live traffic, so this
  is a deliberate checkpoint, not an oversight.
- **Serving** — [FastAPI](https://fastapi.tiangolo.com) + Pydantic. Every
  request is validated against a schema before the handler runs at all.
- **Explainability** — SHAP `TreeExplainer` on the trained LightGBM step,
  surfaced per-prediction as a ranked list of feature contributions, not just
  a global importance chart.

## Project structure

```
credit_scoring/
├── data/
│   ├── raw/                    # DVC-tracked source data
│   └── processed/              # batch-scoring output
├── models/
│   ├── lgbm_pipeline.joblib    # DVC-tracked, currently shipped model
│   └── metrics.json            # metrics of the currently shipped model
├── notebooks/
│   ├── eda.ipynb
│   └── shap_analysis.ipynb     # exploratory SHAP work behind the API's explanations
├── src/credit_scoring/
│   ├── api/                    # FastAPI app: request/response schemas, /predict
│   ├── data/                   # ingestion + Pandera schemas
│   ├── features/               # custom sklearn-compatible transformers
│   ├── models/                 # pipeline construction, Optuna tuning
│   ├── orchestration/          # Prefect tasks/flows: training DAG, batch scoring
│   └── paths.py
├── tests/
├── Dockerfile                  # multi-stage build
├── docker-compose.yml
└── pyproject.toml
```

## Running it

```bash
git clone <repo-url>
cd credit_scoring
dvc pull            # fetches the raw dataset and the trained model
docker compose up --build
```

The API is now live at `http://localhost:8000`. Interactive docs (Swagger UI) at
`http://localhost:8000/docs` — every field, constraint, and example request/response
is generated from the same Pydantic schema the API validates against.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.2,
    "MonthlyIncome": 5400,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
  }'
```

`MonthlyIncome` and `NumberOfDependents` are optional — the trained pipeline has a
`SimpleImputer` for both, matching the fact that not every real applicant reports
income.

```json
{
  "score": 0.07,
  "risk_factors": [
    {"feature": "NumberOfTimes90DaysLate", "contribution": 0.31},
    {"feature": "RevolvingUtilizationOfUnsecuredLines", "contribution": 0.08}
  ]
}
```

### Batch scoring

```bash
python -m credit_scoring.orchestration.batch_score
```

Scores every row of the configured input file through the same registered
pipeline and writes a copy of the input with an added `probability` column.
Input/output paths are currently set in the script's `__main__` block rather
than exposed as CLI arguments — see Limitations.

## Development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest
```

Run the full training DAG locally:

```bash
python src/credit_scoring/orchestration/tasks.py
```

## Metrics

The performance of the currently shipped model lives in `models/metrics.json`
— a small, DVC-independent file that `register()` updates every time it
accepts a new model, so it never goes stale relative to what's actually
deployed. PR-AUC is the metric that governs both Optuna's search and whether a
retrained model is allowed to replace the current one.

## Limitations

Written down honestly, not hidden:

- **Training data is the public "Give Me Some Credit" Kaggle dataset**
  (~150k rows, 2011). Absolute metrics won't transfer to a real lending
  portfolio — the point of this project is the surrounding system
  (contracts, reproducibility, human-gated deployment), not this particular
  dataset's numbers.
- **SHAP output units are not independently verified.** `TreeExplainer`'s
  default output for a gradient-boosted classifier may be in log-odds rather
  than probability space; the *relative ranking* of risk factors is reliable,
  but absolute contribution magnitudes should be read with that caveat until
  confirmed.
- **No drift monitoring.** The system detects "is this new model better than
  the old one at registration time" but not "has the live model's
  performance degraded since it started serving." That's a Phase 5 topic in
  the underlying roadmap.
- **No authentication on `/predict`.** Fine for a local/portfolio deployment,
  not for a service handling real applicant data.
- **Batch scoring paths are hardcoded**, not yet a general-purpose CLI tool.
- **Retraining approval is a single-metric threshold** (PR-AUC not worse),
  with no statistical significance test behind the comparison.

## Tech stack

Python · scikit-learn · LightGBM · Optuna · SHAP · Pandera · FastAPI · Pydantic ·
Prefect · DVC · Docker
