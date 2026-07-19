# 20252B-Gen-AI

Course project for Generative AI (20252B): **Aspect-Based Sentiment Analysis with Generative Report Synthesis**.

Instead of classifying a customer review as simply positive/negative, the system extracts the aspects mentioned (e.g., screen, battery, delivery, price), classifies sentiment per aspect, and uses a generative model to synthesize a short summary report across many reviews — with a lightweight factual checker to guard against the model fabricating stats/insights.

Planned dataset: [SemEval-2014 Task 4 (Laptop Reviews)](https://alt.qcri.org/semeval2014/task4/) for aspect/sentiment labels, with Amazon Reviews for large-scale report generation demos.

See [`docs/Proposal.md`](docs/Proposal.md) for the full proposal and [`docs/Requirements.md`](docs/Requirements.md) for assignment requirements. Progress is tracked in [`plans/project-plan.md`](plans/project-plan.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the official SemEval-2014 Task 4 Laptop XML files from the [Data and Tools page](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) (requires an account) and place them under `data/raw/`, e.g.:

```
data/raw/Laptop_Train_v2.xml
data/raw/Laptops_Test_Gold.xml
```

`data/raw/` is gitignored — the dataset is not redistributed in this repo.

## Baseline

TF-IDF + Logistic Regression baseline for aspect-term polarity classification:

```bash
python scripts/train_baseline.py --train data/raw/Laptop_Train_v2.xml --test data/raw/Laptops_Test_Gold.xml
```

## Tests

```bash
python -m pytest tests/ -v
```

Tests run against a small hand-built fixture (`tests/fixtures/sample_laptop.xml`) that mirrors the official XML schema, so they don't require the gated dataset.
