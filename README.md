# 20252B-Gen-AI

Course project for Generative AI (20252B): **Aspect-Based Sentiment Analysis with Generative Report Synthesis**.

Instead of classifying a customer review as simply positive/negative, the system extracts the aspects mentioned (e.g., screen, battery, delivery, price), classifies sentiment per aspect, and uses a generative model to synthesize a short summary report across many reviews — with a lightweight factual checker to guard against the model fabricating stats/insights.

Dataset: [SemEval-2014 Task 4 (Laptop Reviews)](https://alt.qcri.org/semeval2014/task4/) for aspect/sentiment labels, with Amazon Reviews planned for large-scale report generation demos.

See [`docs/Proposal.md`](docs/Proposal.md) for the full proposal, [`docs/Requirements.md`](docs/Requirements.md) for assignment requirements, and [`docs/Dataset-Verification-Report.md`](docs/Dataset-Verification-Report.md) for how the dataset source was verified. Progress is tracked in [`plans/project-plan.md`](plans/project-plan.md).

## Project Structure

```
.
├── docs/                         # Proposal, requirements, research notes
│   ├── Requirements.md
│   ├── Proposal.md
│   ├── Research-Notes.md               # Tuần 1 findings (ABSA theory, model survey, base repo)
│   ├── Dataset-Options.md              # Alternative datasets considered (reference)
│   ├── Dataset-Verification-Report.md  # Verification of the 3 dataset sources checked
│   ├── Topics.txt / Week0_huy.pdf      # Course-provided material
├── plans/
│   ├── project-plan.md           # Week-by-week plan with progress checkboxes
│   └── task.txt                  # Team task assignment
├── data/
│   ├── raw/                       # Not committed (gitignored) — see data/raw/README.md
│   │   ├── train/                # Laptop_Train_v2.xml (official)
│   │   └── test/                 # Blind PhaseA/PhaseB test files (no gold labels)
│   └── processed/laptop/         # Not committed — train/valid/test split, see data/raw/README.md
├── src/
│   ├── data/
│   │   ├── semeval_loader.py     # Parses SemEval-2014 XML into Sentence/AspectTerm
│   │   └── preprocess.py         # Flattens sentences into (context, aspect, polarity) examples
│   └── baseline/
│       └── tfidf_logreg.py       # TF-IDF + Logistic Regression baseline model
├── scripts/
│   ├── train_baseline.py         # CLI: train + evaluate the baseline
│   └── split_dataset.py          # CLI: split the train XML into train/valid/test XML files
├── notebooks/
│   ├── baseline_semeval_laptop.ipynb            # Self-contained template (Kaggle-ready)
│   └── baseline_semeval_laptop_kaggle_run.ipynb # Executed copy with real results
├── tests/
│   ├── fixtures/sample_laptop.xml  # Hand-built fixture matching the official XML schema
│   └── test_semeval_loader.py
├── results/
│   └── baseline_metrics.json     # Recorded baseline results (Tuần 3)
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the official SemEval-2014 Task 4 Laptop XML files from the [Data and Tools page](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) (requires an account) and place them under `data/raw/`:

```
data/raw/train/Laptop_Train_v2.xml
```

A mirror of the dataset used in this project is also available at [kaggle.com/datasets/dattm03/genai-dataset](https://www.kaggle.com/datasets/dattm03/genai-dataset), for anyone without an account on the official site.

See [`data/raw/README.md`](data/raw/README.md) for what each file is and why there's currently no official gold test file (the baseline holds out a dev split from the train file instead — see below).

`data/raw/` and `data/processed/` are both gitignored — the dataset is not redistributed in this repo. Once `data/raw/train/Laptop_Train_v2.xml` is in place, generate a fixed train/valid/test split with:

```bash
python scripts/split_dataset.py --input data/raw/train/Laptop_Train_v2.xml --output-dir data/processed/laptop
```

## Baseline

TF-IDF + Logistic Regression baseline for aspect-term polarity classification.

No gold test file yet, so by default a 15% dev split is held out from the train file:

```bash
python scripts/train_baseline.py --train data/raw/train/Laptop_Train_v2.xml
```

Once a real gold test XML is available, pass it explicitly:

```bash
python scripts/train_baseline.py --train data/raw/train/Laptop_Train_v2.xml --test data/raw/test/Laptops_Test_Gold.xml
```

A Kaggle-ready, self-contained version of the same baseline is in [`notebooks/baseline_semeval_laptop.ipynb`](notebooks/baseline_semeval_laptop.ipynb) (add the `charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis` Kaggle dataset as input and Run All). The executed copy with actual results is [`notebooks/baseline_semeval_laptop_kaggle_run.ipynb`](notebooks/baseline_semeval_laptop_kaggle_run.ipynb).

**Current result** (dev split, seed=42): Accuracy **0.6211**, Macro-F1 **0.4266** — see [`results/baseline_metrics.json`](results/baseline_metrics.json) and the "Kết Quả Baseline" section in [`docs/Proposal.md`](docs/Proposal.md).

## Tests

```bash
python -m pytest tests/ -v
```

Tests run against a small hand-built fixture (`tests/fixtures/sample_laptop.xml`) that mirrors the official XML schema, so they don't require the gated dataset.
