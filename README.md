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
│   ├── baseline/
│   │   └── tfidf_logreg.py       # TF-IDF + Logistic Regression baseline model
│   └── report/
│       └── aspect_stats.py       # Groups (aspect, sentiment) pairs into a per-aspect summary table
├── scripts/
│   ├── train_baseline.py         # CLI: train + evaluate the baseline
│   └── split_dataset.py          # CLI: split the train XML into train/valid/test XML files
├── notebooks/
│   ├── baseline_semeval_laptop.ipynb              # Baseline template (Kaggle-ready)
│   ├── baseline_semeval_laptop_kaggle_run.ipynb   # Executed baseline, with real results
│   ├── finetune_distilbert_semeval_laptop.ipynb   # Fine-tune DistilBERT (Kaggle-ready)
│   ├── finetune_bert_semeval_laptop.ipynb         # Fine-tune BERT-base (Kaggle-ready)
│   └── aspect_stats_semeval_laptop.ipynb          # Runs the fine-tuned model + aggregates per-aspect stats
├── notebooks-output/              # Executed copies of the notebooks above, with real Kaggle results
│   ├── finetune_distilbert_semeval_laptop_output.ipynb
│   └── aspect_stats_semeval_laptop_output.ipynb
├── tests/
│   ├── fixtures/sample_laptop.xml  # Hand-built fixture matching the official XML schema
│   ├── test_semeval_loader.py
│   └── test_aspect_stats.py
├── results/
│   └── baseline_metrics.json     # Recorded baseline results (Tuần 3)
├── output/
│   └── aspect_stats.txt          # Aggregated per-aspect sentiment stats (gold + predicted), from the Kaggle run
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

## Fine-tuning DistilBERT/BERT

Same task as the baseline (aspect-term polarity classification), fine-tuned as a sentence-pair
classifier: `[CLS] sentence [SEP] aspect_term [SEP]`. 3 labels — `positive`/`negative`/`neutral`
(`conflict` was dropped from the dataset, too few examples to learn reliably).

Both notebooks are Kaggle-ready and self-contained:

- [`notebooks/finetune_distilbert_semeval_laptop.ipynb`](notebooks/finetune_distilbert_semeval_laptop.ipynb) — `distilbert-base-uncased`, lr=3e-5.
- [`notebooks/finetune_bert_semeval_laptop.ipynb`](notebooks/finetune_bert_semeval_laptop.ipynb) — `bert-base-uncased`, lr=2e-5.

Each trains 3 seeds (42/43/44) × up to 20 epochs with early stopping (patience=3 on valid
macro-F1), class-weighted cross-entropy loss to counter the `positive`/`negative` vs `neutral`
imbalance, then reports mean ± std accuracy/macro-F1 over the 3 seeds on the held-out test split
and saves the best-of-3-seeds model.

**Kaggle setup**: add the [`dattm03/genai-dataset`](https://www.kaggle.com/datasets/dattm03/genai-dataset)
dataset as input, enable a GPU accelerator, Run All.

**Current result (DistilBERT)**: Accuracy **0.7447 ± 0.0159**, Macro-F1 **0.6861 ± 0.0235** (best
seed: 44) — see the executed notebook
[`notebooks-output/finetune_distilbert_semeval_laptop_output.ipynb`](notebooks-output/finetune_distilbert_semeval_laptop_output.ipynb)
and `plans/project-plan.md` (Tuần 4) for the full per-label breakdown. BERT-base result pending.

After a run finishes, the fine-tuned model (`{distilbert,bert}-absa-model/`, ~270-420MB) is saved
to `/kaggle/working/` — download it from the notebook's Output tab and upload it as its own Kaggle
Dataset or Model to reuse in the aspect-stats notebook below (pick **PyTorch** as the framework if
using Kaggle's Models feature). If you create it via "New Model from notebook output" it also pulls
in the per-seed training checkpoints (`{...}-absa-seed<N>/`, several hundred MB each) — harmless
clutter, `aspect_stats_semeval_laptop.ipynb` knows to skip them and only use the `*-absa-model` folder.

## Aspect-level statistics

Turns per-`(aspect, sentiment)` example pairs into a per-aspect summary table
(`positive`/`negative`/`neutral` counts + majority sentiment) — the input the FLAN-T5
report-generation step (Tuần 4) consumes.

- [`src/report/aspect_stats.py`](src/report/aspect_stats.py) — the aggregation logic itself, pure
  Python (no GPU/transformers needed), runs and is tested locally
  ([`tests/test_aspect_stats.py`](tests/test_aspect_stats.py)).
- [`notebooks/aspect_stats_semeval_laptop.ipynb`](notebooks/aspect_stats_semeval_laptop.ipynb) —
  the Kaggle side: loads the fine-tuned model, runs inference over every `(sentence, aspect)`
  example in the dataset, and aggregates both the gold-label table and the model-predicted table
  (the one that generalizes to unlabeled data) for comparison.

**Kaggle setup**: add `dattm03/genai-dataset` **and** the fine-tuned-model dataset/model created
above as inputs, Run All (no GPU required — inference-only on a small dataset).

**Current result**: 253 aspects with ≥2 mentions; majority-sentiment agreement between the gold
and predicted tables **90.12%**. Full table:
[`output/aspect_stats.txt`](output/aspect_stats.txt); executed notebook:
[`notebooks-output/aspect_stats_semeval_laptop_output.ipynb`](notebooks-output/aspect_stats_semeval_laptop_output.ipynb).

## Tests

```bash
python -m pytest tests/ -v
```

Tests run against a small hand-built fixture (`tests/fixtures/sample_laptop.xml`, for `test_semeval_loader.py`) and synthetic data (`test_aspect_stats.py`) — neither needs the gated dataset or GPU/transformers.

## FLAN-T5 report generation and factual checking

The end-to-end pipeline reads the real DistilBERT statistics in
[`output/aspect_stats.txt`](output/aspect_stats.txt). It uses the `predicted` table by default,
because this is the table available in a real deployment where gold sentiment labels do not exist.
After installing the requirements, run:

```bash
python scripts/generate_report.py --output output/flan_t5_report.json
```

This is equivalent to explicitly running:

```bash
python scripts/generate_report.py --stats output/aspect_stats.txt --table predicted \
  --model google/flan-t5-base --max-aspects 4 --max-attempts 3 \
  --output output/flan_t5_report.json
```

The default model is `google/flan-t5-base` with deterministic beam search. The pipeline selects a
popular aspect, a positive strength, a negative weakness, and a divided aspect. A strict few-shot
copy-transformation prompt explains that all values are mention counts—not prices or product
specifications—and asks for one auditable sentence per row. The prompt asks for an
auditable analytical sentences such as `Screen attracted the most attention, with 60 mentions: 32
positive, 24 negative, and 4 neutral.` The input explicitly labels the most-discussed aspect,
clearest strength, main concern, and most-divided feedback so FLAN-T5 can explain insights without
inventing them. The
checker maps each claim to its source aspect, verifies all four counts, rejects unknown aspects,
missing/duplicate aspects, extra prose, and numbers outside the auditable format. Failed generations
are retried up to three times without reloading the model. Every candidate in `attempt_history` is
an actual FLAN-T5 output. `accepted` is true only when every selected aspect is present and the
complete report passes the checker. If every attempt fails, the final FLAN-T5 output is returned
with `accepted: false`; the program never replaces it with a template-generated report.

To check an existing report without running FLAN-T5 again:

```bash
python scripts/generate_report.py --stats output/aspect_stats.txt --table predicted \
  --report-file path/to/report.txt
```

Implementation: `src/report/flan_t5_report.py`, `src/report/factual_checker.py`, and
`src/report/stats_io.py`. Tests are in `tests/test_report_generation.py`.

The fine-tuned model can be produced with the self-contained Colab notebook
`notebooks/finetune_flan_t5_report_colab.ipynb` (upload only `output/aspect_stats.txt`).

The notebook prevents evaluation leakage: train, validation, and synthetic test use three mutually
disjoint synthetic aspect-name pools and independently generated counts. All real `predicted`
aspect names and counts from `aspect_stats.txt` are reserved exclusively for the final evaluation;
runtime assertions verify both aspect-level and example-level disjointness.

After extracting the exported model to `models/flan-t5-report`, run:

```bash
python scripts/generate_report.py --model models/flan-t5-report --max-aspects 4 \
  --max-attempts 1 --output output/flan_t5_finetuned_report.json
```

Use the same decoding configuration as the Colab evaluation: deterministic four-beam generation
without repetition penalties. The recorded real-table run passed all 4/4 claims.
