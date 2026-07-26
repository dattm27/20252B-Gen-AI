========================================================================
20252B-Gen-AI
Aspect-Based Sentiment Analysis with Generative Report Synthesis
========================================================================

Course project for Generative AI (20252B).

Instead of classifying a customer review as simply positive/negative, this
system extracts the aspects mentioned in a review (e.g. screen, battery,
food, service), classifies sentiment per aspect, and uses a generative
model to synthesize a short summary report across many reviews -- with a
lightweight factual checker to guard against the model fabricating
stats/insights.

Two tracks were built, on two datasets:

  1. Laptop domain (SemEval-2014 Task 4): given a sentence and a known
     aspect term, classify its sentiment. Baseline -> DistilBERT -> BERT.
  2. Restaurant domain (SemEval Triplet data): given a raw sentence with
     no gold aspect, generate (aspect, opinion, sentiment) triplets
     directly. Baseline -> T5-small -> T5-base -> FLAN-T5-base. The
     "opinion" field doubles as a quotable reason for the report.

Full narrative, all results tables, and design decisions are in
plans/project-plan.md (progress log) and README.md (technical reference
with links to every notebook/script/result file). This file only covers
installation and how to run each piece.


------------------------------------------------------------------------
1. REQUIREMENTS
------------------------------------------------------------------------

- Python 3.9+
- Everything in requirements.txt (scikit-learn, numpy, scipy, pytest) --
  deliberately lightweight. All the BERT/T5 fine-tuning and inference
  runs on Kaggle notebooks (free GPU), not locally, so this repo's local
  environment never needs to install torch/transformers.


------------------------------------------------------------------------
2. INSTALL
------------------------------------------------------------------------

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt


------------------------------------------------------------------------
3. GET THE DATA
------------------------------------------------------------------------

3a. Laptop domain (SemEval-2014 Task 4)

Download the official Laptop_Train_v2.xml from:
    https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
(requires a free account), and place it at:
    data/raw/train/Laptop_Train_v2.xml

A mirror is also available on Kaggle (no account needed):
    https://www.kaggle.com/datasets/dattm03/genai-dataset

Then generate the fixed train/valid/test split used by every notebook:

    python scripts/split_dataset.py \
        --input data/raw/train/Laptop_Train_v2.xml \
        --output-dir data/processed/laptop

3b. Restaurant domain (SemEval Triplet data) -- only needed for the local
    ASTE baseline (section 5b below); the Kaggle notebooks fetch this
    automatically.

    git clone --depth 1 https://github.com/xuuuluuu/SemEval-Triplet-data.git \
        data/raw/semeval-triplet-data

Note: data/raw/ and data/processed/ are both gitignored -- datasets are
not redistributed in this repo, you must download them yourself.


------------------------------------------------------------------------
4. RUN THE TESTS
------------------------------------------------------------------------

    python -m pytest tests/ -v

29 tests, all run against small hand-built fixtures / synthetic data --
no GPU, no downloaded dataset required.


------------------------------------------------------------------------
5. RUN LOCALLY (no GPU needed)
------------------------------------------------------------------------

5a. Laptop baseline (TF-IDF + Logistic Regression)

    python scripts/train_baseline.py \
        --train data/processed/laptop/train.xml \
        --test data/processed/laptop/test.xml

5b. Restaurant/ASTE baseline (frequency lookup, no gold aspect needed)

    python scripts/eval_aste_baseline.py \
        --data-dir data/raw/semeval-triplet-data/ASTE-Data-V1-AAAI2020

Both print Accuracy/Macro-F1 (5a) or Precision/Recall/Triplet-F1 (5b) to
the console and save a metrics JSON under results/.


------------------------------------------------------------------------
6. RUN ON KAGGLE (GPU required for fine-tuning/inference)
------------------------------------------------------------------------

Every notebook under notebooks/ is self-contained and Kaggle-ready: open
it on Kaggle, add the listed dataset(s)/model(s) as Input, enable a GPU
accelerator, and Run All. Already-executed copies with real results are
under notebooks-output/, for reference without re-running anything.

Run them in this order (each step's output feeds the next):

  1. notebooks/finetune_distilbert_semeval_laptop.ipynb
     notebooks/finetune_bert_semeval_laptop.ipynb
       -> fine-tune the Laptop sentiment classifiers.
       Kaggle input: dataset dattm03/genai-dataset.

  2. notebooks/aspect_stats_semeval_laptop.ipynb
       -> run the fine-tuned BERT model + aggregate per-aspect stats.
       Kaggle input: dattm03/genai-dataset + the fine-tuned model
       (download {distilbert,bert}-absa-model/ from step 1's Output tab,
       upload as a new Kaggle Dataset/Model, framework "PyTorch").

  3. notebooks/train-t5-small-for-aste-on-14res-15res-16res.ipynb
     notebooks/train-t5-base-for-aste-on-14res-15res-16res.ipynb
     notebooks/train-flan-t5-base-for-aste-on-14res-15res-16res.ipynb
       -> fine-tune the 3 ASTE models being compared (Restaurant domain).
       Kaggle input: none required (auto-clones the dataset); enable
       Internet + GPU.

  4. notebooks/aste_aspect_reasons_restaurant.ipynb
       -> run the chosen model (t5-base) + aggregate aspect + top-10
       reason phrases, with a gold-vs-predicted sanity check.
       Kaggle input: the t5-base-aste-restaurant-best/ model from step 3
       (upload the same way as step 2).

  5. notebooks/aste_aspect_reasons_yelp_demo.ipynb
       -> same as step 4, at scale, on real unlabeled Yelp reviews (no
       benchmark, demo only).
       Kaggle input: dataset farukalam/yelp-restaurant-reviews + the
       same t5-base model as step 4.

Each notebook's own top markdown cell repeats its exact Kaggle input
requirements.


------------------------------------------------------------------------
7. WHERE RESULTS LIVE
------------------------------------------------------------------------

  results/*.json    Baseline metrics (Laptop + Restaurant tracks)
  output/*.json      Per-aspect sentiment/reason tables (final report
                     input) -- .txt for the older Laptop-track file
  plans/project-plan.md   Full week-by-week log with every result table
                          and the reasoning behind each design decision
  README.md          Same as this file, with full narrative, all result
                     tables, and links to every notebook/script


------------------------------------------------------------------------
8. PROJECT LAYOUT
------------------------------------------------------------------------

  docs/         Proposal, requirements, research notes
  plans/        Week-by-week plan (project-plan.md) + task assignment
  data/         Not committed -- see section 3 above
  src/          Reusable, tested Python modules
    data/         XML/triplet-format parsers
    baseline/     Non-neural baselines (TF-IDF+LogReg, ASTE lookup)
    report/       Per-aspect aggregation logic
  scripts/      Local CLIs (baseline training/eval)
  notebooks/    Kaggle-ready notebook templates
  notebooks-output/  Executed copies with real results
  tests/        pytest suite (29 tests, all local)
  results/      Recorded metrics JSON
  output/       Final per-aspect stats/reasons tables
