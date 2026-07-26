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
│   │   ├── preprocess.py         # Flattens sentences into (context, aspect, polarity) examples
│   │   └── aste_loader.py        # Parses the ASTE triplet tag format (aspect, opinion, sentiment)
│   ├── baseline/
│   │   ├── tfidf_logreg.py            # TF-IDF + Logistic Regression baseline (Laptop classification)
│   │   └── aste_lookup_baseline.py    # Frequency-lookup baseline (Restaurant ASTE — no known aspect needed)
│   └── report/
│       └── aspect_stats.py       # Per-aspect summary tables: sentiment counts, + top-N reason phrases
├── scripts/
│   ├── train_baseline.py         # CLI: train + evaluate the Laptop baseline
│   ├── eval_aste_baseline.py     # CLI: train + evaluate the ASTE lookup baseline (local, no GPU)
│   └── split_dataset.py          # CLI: split the train XML into train/valid/test XML files
├── notebooks/
│   ├── baseline_semeval_laptop.ipynb              # Baseline template (Kaggle-ready)
│   ├── baseline_semeval_laptop_kaggle_run.ipynb   # Executed baseline, with real results
│   ├── finetune_distilbert_semeval_laptop.ipynb   # Fine-tune DistilBERT (Kaggle-ready)
│   ├── finetune_bert_semeval_laptop.ipynb         # Fine-tune BERT-base (Kaggle-ready)
│   ├── aspect_stats_semeval_laptop.ipynb          # Runs the fine-tuned model + aggregates per-aspect stats
│   ├── train-t5-small-for-aste-on-14res-15res-16res.ipynb      # ASTE: fine-tune t5-small (Kaggle-ready)
│   ├── train-t5-base-for-aste-on-14res-15res-16res.ipynb       # ASTE: fine-tune t5-base (Kaggle-ready)
│   ├── train-flan-t5-base-for-aste-on-14res-15res-16res.ipynb  # ASTE: fine-tune flan-t5-base (Kaggle-ready)
│   ├── aste_aspect_reasons_restaurant.ipynb                    # ASTE: t5-base inference + aspect/reasons aggregation (Kaggle-ready)
│   └── aste_aspect_reasons_yelp_demo.ipynb                     # ASTE: same, large-scale demo on unlabeled Yelp reviews (Kaggle-ready, Tuần 5)
├── notebooks-output/              # Executed copies of the notebooks above, with real Kaggle results
│   ├── finetune_distilbert_semeval_laptop_output.ipynb
│   ├── aspect_stats_semeval_laptop_output.ipynb
│   ├── train-t5-small-for-aste-on-14res-15res-16res-output.ipynb
│   ├── train-t5-base-for-aste-on-14res-15res-16res-output.ipynb
│   ├── train-flan-t5-base-for-aste-on-14res-15res-16r-output.ipynb
│   ├── train-flan-t5-base-for-aste-on-14res-15res-16r-1-e-4.ipynb
│   └── aste_aspect_reasons_yelp_demo_output.ipynb
├── tests/
│   ├── fixtures/sample_laptop.xml  # Hand-built fixture matching the official XML schema
│   ├── test_semeval_loader.py
│   ├── test_aspect_stats.py
│   ├── test_aste_loader.py
│   └── test_aste_lookup_baseline.py
├── results/
│   ├── baseline_metrics.json          # Recorded baseline results (Tuần 3, 4-class dataset — historical)
│   ├── baseline_metrics_3class.json   # Baseline re-run on the current 3-class split, for a fair BERT comparison (Tuần 5)
│   └── aste_baseline_metrics.json     # AsteLookupBaseline results (Tuần 5)
├── output/
│   ├── aspect_stats.txt                    # Per-aspect sentiment stats (Laptop/BERT track, gold + predicted)
│   ├── aspect_reasons_restaurant.json      # Per-aspect sentiment + top-10 reasons (Restaurant/t5-base ASTE track)
│   └── aspect_reasons_yelp_demo.json       # Same, large-scale demo on unlabeled Yelp Restaurant Reviews (Tuần 5)
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

**Current result** (dev split, seed=42): Accuracy **0.6211**, Macro-F1 **0.4266** — see [`results/baseline_metrics.json`](results/baseline_metrics.json) and the "Kết Quả Baseline" section in [`docs/Proposal.md`](docs/Proposal.md). That run predates the dataset dropping the `conflict` label; for a fair comparison against the 3-label DistilBERT/BERT results below, the baseline was re-run on the same `data/processed/laptop/{train,test}.xml` split (`python scripts/train_baseline.py --train data/processed/laptop/train.xml --test data/processed/laptop/test.xml`): Accuracy **0.6171**, Macro-F1 **0.5581** — see [`results/baseline_metrics_3class.json`](results/baseline_metrics_3class.json). Macro-F1 jumps a lot just from dropping `conflict` (a class the baseline never got right, which tanks an unweighted macro average even at <0.3% of the data) — the original 4-class number isn't a fair reference point for the BERT comparison below.

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

**Current results**:

| | Accuracy | Macro-F1 | best seed |
|---|---|---|---|
| DistilBERT | 0.7447 ± 0.0159 | 0.6861 ± 0.0235 | 44 |
| BERT-base | **0.7627 ± 0.0239** | **0.7123 ± 0.0324** | 44 |

BERT-base wins on every label, most notably `neutral` (F1 0.502 vs 0.458) — it's the model used
downstream (aspect stats, report generation). See the executed notebooks
([`notebooks-output/finetune_distilbert_semeval_laptop_output.ipynb`](notebooks-output/finetune_distilbert_semeval_laptop_output.ipynb),
[`notebooks-output/finetune_bert_semeval_laptop_output.ipynb`](notebooks-output/finetune_bert_semeval_laptop_output.ipynb))
and `plans/project-plan.md` (Tuần 4) for the full per-label breakdown.

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

**Current results**: 253 aspects with ≥2 mentions in both runs; majority-sentiment agreement
between the gold and predicted tables:

| Model | Agreement | Full table | Executed notebook |
|---|---|---|---|
| DistilBERT | 90.12% | [`output/aspect_stats.txt`](output/aspect_stats.txt) | [`notebooks-output/aspect_stats_semeval_laptop_output.ipynb`](notebooks-output/aspect_stats_semeval_laptop_output.ipynb) |
| **BERT-base** | **93.28%** | [`output/aspect_stats_bert.txt`](output/aspect_stats_bert.txt) | [`notebooks-output/aspect-level-sentiment-statistics-bert-output.ipynb`](notebooks-output/aspect-level-sentiment-statistics-bert-output.ipynb) |

`output/aspect_stats_bert.txt` is the official table handed off to report generation — the
DistilBERT run is kept only for comparison.

## Aspect Sentiment Triplet Extraction (ASTE)

A parallel, additive track (not a replacement for the DistilBERT/BERT classification track
above): a T5-family seq2seq model reads a raw sentence and directly generates
`aspect: X | opinion: Y | sentiment: Z` triplets — no gold aspect term needed, and the `opinion`
span doubles as a **reason/rationale** the report-generation step can quote (e.g. "service is
*poor*"), instead of a bare aspect+sentiment count.

- Dataset: [SemEval Triplet data](https://github.com/xuuuluuu/SemEval-Triplet-data), **restaurant
  domain** (14res/15res/16res) — different domain from the Laptop data used above.
- Notebooks self-locate the dataset: first checks `semi-triple-{14,15,16}res` Kaggle inputs, falls
  back to cloning the GitHub repo if not found (enable internet + GPU on Kaggle).
- Three models compared under identical hyperparameters (20 epochs, lr=3e-4, effective batch 16)
  so only the checkpoint changes:
  [`notebooks/train-t5-small-for-aste-on-14res-15res-16res.ipynb`](notebooks/train-t5-small-for-aste-on-14res-15res-16res.ipynb),
  [`notebooks/train-t5-base-for-aste-on-14res-15res-16res.ipynb`](notebooks/train-t5-base-for-aste-on-14res-15res-16res.ipynb),
  [`notebooks/train-flan-t5-base-for-aste-on-14res-15res-16res.ipynb`](notebooks/train-flan-t5-base-for-aste-on-14res-15res-16res.ipynb).

**Current results** (test split, triplet-level F1 = set overlap of predicted vs gold
`(aspect, opinion, sentiment)` triples):

| Model | Test triplet-F1 | Test P / R | Test exact match | Executed notebook |
|---|---|---|---|---|
| [`AsteLookupBaseline`](src/baseline/aste_lookup_baseline.py) (non-neural) | 0.3736 | 0.3222 / 0.4446 | — | `python scripts/eval_aste_baseline.py` (local) |
| t5-small | 0.7240 | 0.7238 / 0.7243 | 0.6358 | [`notebooks-output/train-t5-small-for-aste-on-14res-15res-16res-output.ipynb`](notebooks-output/train-t5-small-for-aste-on-14res-15res-16res-output.ipynb) |
| **t5-base** | **0.7442** | 0.7609 / 0.7282 | **0.6481** | [`notebooks-output/train-t5-base-for-aste-on-14res-15res-16res-output.ipynb`](notebooks-output/train-t5-base-for-aste-on-14res-15res-16res-output.ipynb) |
| flan-t5-base (lr=3e-4) | 0.5898 | 0.5910 / 0.5887 | 0.4877 | [`notebooks-output/train-flan-t5-base-for-aste-on-14res-15res-16r-output.ipynb`](notebooks-output/train-flan-t5-base-for-aste-on-14res-15res-16r-output.ipynb) |
| flan-t5-base (lr=1e-4, retry) | 0.4159 | 0.4185 / 0.4133 | 0.3210 | [`notebooks-output/train-flan-t5-base-for-aste-on-14res-15res-16r-1-e-4.ipynb`](notebooks-output/train-flan-t5-base-for-aste-on-14res-15res-16r-1-e-4.ipynb) |

`AsteLookupBaseline` is the non-neural reference point for this track (the Laptop track's
TF-IDF+LogReg baseline can't do ASTE — it needs a known aspect term up front). No training beyond
frequency counting: it memorizes aspect phrases and each opinion phrase's majority sentiment from
train, then at test time looks for exact phrase matches and pairs each aspect with its nearest
opinion match. Evaluated with the same corpus-level triplet-F1 (`src/data/aste_loader.py::corpus_triplet_prf`,
tested in `tests/test_aste_loader.py`) on the real `ASTE-Data-V1-AAAI2020` split (2735 train /
1134 test sentences, matching the T5 notebooks exactly) via `scripts/eval_aste_baseline.py` — runs
locally, no GPU. t5-base nearly doubles its triplet-F1 (0.3736 → 0.7442); the baseline's weak
recall (0.4446) is expected — any aspect/opinion phrased differently than training is missed
entirely, which is exactly the kind of generalization a fine-tuned model provides.

**t5-base wins on every metric — it's the chosen model** for this track. flan-t5-base was tried
at two learning rates and both lose decisively, so no further tuning was invested. Notably, the
initial "lr=3e-4 is too high" hypothesis was **wrong**: dropping to `lr=1e-4` made results *worse*
(0.4159 vs 0.5898), and its training log shows `eval_triplet_f1` still climbing steadily through
epoch 20 (0.244 → 0.470) with no sign of plateauing — the lower lr just converges slower, it
doesn't fix anything. The likely real explanation: flan-t5-base (already instruction-tuned across
many diverse tasks) needs substantially more epochs to re-specialize on this terse structured
extraction format, not a different lr — a useful cautionary finding for the report (don't
conclude root cause from tuning a single hyperparameter).

## Aspect + top-10-reasons (t5-base)

Aggregates t5-base's `(aspect, opinion, sentiment)` triplets into one row per aspect —
`positive`/`negative`/`neutral` counts, plus the top-10 most frequent opinion phrases ("reasons")
per sentiment — the input the FLAN-T5 report-generation step consumes (richer than the
Laptop/BERT track's plain aspect+sentiment table, since it now has quotable reasons).

- [`src/data/aste_loader.py`](src/data/aste_loader.py) / [`src/report/aspect_stats.py`](src/report/aspect_stats.py)
  (`aggregate_aspect_reasons`) — the parsing/aggregation logic, pure Python, tested locally
  ([`tests/test_aste_loader.py`](tests/test_aste_loader.py), [`tests/test_aspect_stats.py`](tests/test_aspect_stats.py)).
- [`notebooks/aste_aspect_reasons_restaurant.ipynb`](notebooks/aste_aspect_reasons_restaurant.ipynb) —
  the Kaggle side: loads t5-base, runs inference over the full dataset, and inlines the same
  tested parsing/aggregation code to produce gold + predicted tables.

  Originally planned to run fully locally (no Kaggle round-trip, so it chains straight into
  report generation) — dropped after t5-base beam-search inference over ~4500 sentences was
  estimated at ~3-3.5 hours on CPU. Only the model inference moved back to Kaggle GPU; the
  aggregation logic stayed local/tested and is simply inlined into the notebook.

**Kaggle setup**: add the t5-base ASTE model (`t5-base-aste-restaurant-best/`, from the training
notebook above) as a Kaggle Dataset/Model input, GPU accelerator.

**Current result**: 4550 sentences (train+dev+test, all 3 domains); 619 gold / 585 predicted
aspects with ≥2 mentions; majority-sentiment agreement **531/546 = 97.25%** (higher than the
Laptop/BERT track's 93.28% — the model predicts aspect, opinion, and sentiment jointly instead of
sentiment for an already-known aspect). Example row (`food`, 827 mentions, majority positive):
positive reasons `great`(109)/`good`(100)/`delicious`(39)/...; negative reasons
`mediocre`(10)/`bad`(7)/`overpriced`(5)/... Full table:
[`output/aspect_reasons_restaurant.json`](output/aspect_reasons_restaurant.json).

## Large-scale demo (Yelp Restaurant Reviews, Tuần 5)

Same pipeline as above, pointed at real, **unlabeled** reviews instead of the labeled SemEval set
— the Tuần 5 "test report generation at scale" requirement. Originally planned to use Amazon
Reviews (`docs/Proposal.md`), swapped to [Yelp Restaurant
Reviews](https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews): Amazon Reviews
(Electronics category) is a different domain than what t5-base was fine-tuned on (restaurant
triplets) and would likely produce meaningless extractions, while Yelp is the same domain.

[`notebooks/aste_aspect_reasons_yelp_demo.ipynb`](notebooks/aste_aspect_reasons_yelp_demo.ipynb) —
adds one extra step versus the notebook above: real Yelp reviews are full multi-sentence
paragraphs (the model was fine-tuned on single short sentences), so each sampled review is split
into sentences (simple regex heuristic, no new NLP dependency) before inference. No gold labels,
so only a `predicted` table is produced — trustworthiness rests on the 97.25% agreement already
validated on labeled same-domain data above.

**Kaggle setup**: add `farukalam/yelp-restaurant-reviews` and the t5-base ASTE model as inputs,
GPU accelerator. `SAMPLE_SIZE` (default 2000 reviews) caps runtime for a first pass — raise it for
a bigger demo.

**Current result**: 2000 sampled reviews (seed=42) → 14285 sentences (7.1/review) → 14700
triplets → 1224 aspects with ≥2 mentions. The sample skewed toward dessert/bakery shops: top
aspects `ice cream` (765 mentions, positive), `place` (633), `flavors` (237), plus
`staff`/`service`/`donuts`/`pastries`/`bakery`/`cookies`/`macarons` — plausible, on-domain
results. Some extraction noise is visible too (e.g. `extract`, 196 mentions — checked its reason phrases
in error analysis; they're generic/incoherent (`good`, `worth`, `try`, `4.`, `huh`), not
consistently about "vanilla extract" as first guessed — more likely the model latching onto a
stray token on short/fragmentary sentences after sentence-splitting). Full error analysis:
`report/main.md` §5.5 / `plans/project-plan.md` Tuần 5. Executed notebook:
[`notebooks-output/aste_aspect_reasons_yelp_demo_output.ipynb`](notebooks-output/aste_aspect_reasons_yelp_demo_output.ipynb);
full table: [`output/aspect_reasons_yelp_demo.json`](output/aspect_reasons_yelp_demo.json).

## Tests

```bash
python -m pytest tests/ -v
```

Tests run against a small hand-built fixture (`tests/fixtures/sample_laptop.xml`, for `test_semeval_loader.py`) and synthetic/hand-written data (`test_aspect_stats.py`, `test_aste_loader.py`, `test_aste_lookup_baseline.py`, `test_report_generation.py`) — none need the gated dataset or GPU/transformers.

## FLAN-T5 report generation and factual checking

### Current reason-aware restaurant pipeline

The current input is
[`output/aspect_reasons_restaurant.json`](output/aspect_reasons_restaurant.json), using its
`predicted` table. Each aspect contains sentiment counts plus frequent positive, negative, and
neutral reason phrases.

Fine-tune with the self-contained Colab notebook
[`notebooks/finetune_flan_t5_reasoned_report_colab.ipynb`](notebooks/finetune_flan_t5_reasoned_report_colab.ipynb).
Upload only the reason JSON, run all cells, download the model ZIP, and extract it to
`models/flan-t5-reasoned-report`. Then run:

```bash
python scripts/generate_report.py \
  --model models/flan-t5-reasoned-report \
  --max-aspects 4 \
  --max-attempts 3 \
  --output output/flan_t5_reasoned_report.json
```

The selected input rows do not carry preassigned role labels. FLAN-T5 compares them and writes a
connected paragraph grounded in exact counts and supplied reasons. The checker is not tied to one
exact sentence template: it associates sentences with aspects, rejects unsupported numbers, and
requires count and reason evidence for each selected aspect. There is no fallback report.

The real restaurant table is reserved for final evaluation only. Synthetic train, validation, and
test splits have disjoint aspect vocabularies, independently generated counts, shuffled input rows,
and several target discourse organizations. Human-written or carefully reviewed target reports
would be the next step beyond this synthetic baseline.

The laptop instructions below describe the older fixed-template experiment retained for comparison.

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
