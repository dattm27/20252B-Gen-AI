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

Both tracks feed the same final step: a fine-tuned FLAN-T5 model turns
the per-aspect statistics into a short natural-language report, checked
by a rule-based factual checker before being accepted.

Full narrative, all results tables, and design decisions are in
plans/project-plan.md (progress log) and README.md (technical reference
with links to every notebook/script/result file). This file only covers
installation and how to run each piece.


------------------------------------------------------------------------
1. REQUIREMENTS
------------------------------------------------------------------------

- Python 3.9+
- Everything in requirements.txt:
    scikit-learn / numpy / scipy / pytest   -- lightweight local baselines
    transformers / torch / sentencepiece / protobuf
        -- needed locally to run FLAN-T5 report generation (section 5c)
           and the local ASTE + report demo (section 5d). CPU is enough
           for both; no GPU required. All BERT/T5 fine-tuning still
           happens on Kaggle/Colab notebooks (free GPU) -- only
           inference/generation runs locally.
  transformers is pinned to <4.50: 4.57+ fails to load these T5
  checkpoints' tokenizers (a regression in how it reads older
  extra_special_tokens config format).


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

58 tests, all run against small hand-built fixtures / synthetic data --
no GPU, no downloaded dataset, no model checkpoint required.


------------------------------------------------------------------------
5. RUN LOCALLY
------------------------------------------------------------------------

5a. Laptop baseline (TF-IDF + Logistic Regression, no GPU)

    python scripts/train_baseline.py \
        --train data/processed/laptop/train.xml \
        --test data/processed/laptop/test.xml

5b. Restaurant/ASTE baseline (frequency lookup, no gold aspect needed, no GPU)

    python scripts/eval_aste_baseline.py \
        --data-dir data/raw/semeval-triplet-data/ASTE-Data-V1-AAAI2020

Both print Accuracy/Macro-F1 (5a) or Precision/Recall/Triplet-F1 (5b) to
the console and save a metrics JSON under results/.

5c. FLAN-T5 report generation (needs transformers/torch, CPU is fine)

Turns a per-aspect stats table into a short report and checks it against
the source numbers/reasons:

    python scripts/generate_report.py \
        --stats output/aspect_stats.txt --table predicted \
        --model google/flan-t5-base \
        --max-aspects 4 --max-attempts 3 \
        --output output/flan_t5_report.json

With no --model, this downloads google/flan-t5-base from the HF Hub the
first time (zero-shot, no fine-tuned checkpoint needed). To use a
fine-tuned checkpoint instead, download it from the matching Colab
notebook (see section 6) and point --model at the extracted folder, e.g.
--model models/flan-t5-reasoned-report.

To check an already-written report instead of generating one:

    python scripts/generate_report.py --stats output/aspect_stats.txt \
        --table predicted --report-file path/to/report.txt

5d. Local ASTE + report demo (small sample, CPU, no Kaggle/Colab round-trip)

Runs the full ASTE -> aggregation -> FLAN-T5 report pipeline end-to-end on
a small local sample of restaurant sentences (data/demo/sample_reviews.json,
10 sentences covering positive/negative/mixed/neutral/negation/sarcasm
cases). Needs the two checkpoints below downloaded and placed under
models/ first (see section 6 for where each comes from):

    models/t5-base-aste-restaurant-best/
    models/flan-t5-reasoned-report/

Then:

    python scripts/infer_aste_demo.py \
        --model models/t5-base-aste-restaurant-best \
        --input data/demo/sample_reviews.json \
        --output output/aspect_reasons_demo.json

    python scripts/generate_report.py \
        --stats output/aspect_reasons_demo.json --table predicted \
        --model models/flan-t5-reasoned-report \
        --max-aspects 4 --max-attempts 3 \
        --output output/flan_t5_demo_report.json

~10 sentences takes well under a minute total on a laptop CPU. With only
~10 source sentences, most aspects have just 1-3 mentions -- far sparser
than the 55-180 mentions/aspect the model was fine-tuned on -- so the
checker often (correctly) rejects the report for a repeated or missing
aspect. That rejection is itself part of the demo: it shows the factual
checker catching under-grounded output instead of passing it through.


------------------------------------------------------------------------
6. RUN ON KAGGLE / COLAB (GPU required for fine-tuning/inference)
------------------------------------------------------------------------

Every notebook under notebooks/ is self-contained and Kaggle-ready: open
it on Kaggle, add the listed dataset(s)/model(s) as Input, enable a GPU
accelerator, and Run All. Already-executed copies with real results are
under notebooks-output/, for reference without re-running anything. The
two FLAN-T5 notebooks below are Colab notebooks instead (free T4 GPU,
no dataset-upload step needed -- just upload one JSON/txt file).

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
       (upload the same way as step 2). Download this model's Output
       folder to use locally too (section 5d).

  5. notebooks/aste_aspect_reasons_yelp_demo.ipynb
       -> same as step 4, at scale, on real unlabeled Yelp reviews (no
       benchmark, demo only).
       Kaggle input: dataset farukalam/yelp-restaurant-reviews + the
       same t5-base model as step 4.

  6. notebooks/finetune_flan_t5_report_colab.ipynb
     notebooks/finetune_flan_t5_reasoned_report_colab.ipynb
       -> fine-tune FLAN-T5 to turn a per-aspect stats table into a
       checked natural-language report (fixed-template vs reason-aware
       variants -- the reason-aware one is the current pipeline).
       Colab: upload output/aspect_stats.txt (fixed-template variant) or
       output/aspect_reasons_restaurant.json (reason-aware variant),
       Run All, download the exported model ZIP. Extract it to
       models/flan-t5-report or models/flan-t5-reasoned-report and run
       it with scripts/generate_report.py (section 5c/5d).

Each notebook's own top markdown cell repeats its exact Kaggle/Colab
input requirements.


------------------------------------------------------------------------
7. WHERE RESULTS LIVE
------------------------------------------------------------------------

  results/*.json    Baseline metrics (Laptop + Restaurant tracks)
  output/*.json      Per-aspect sentiment/reason tables + generated
                     reports (final report input/output) -- .txt for the
                     older Laptop-track stats file
  models/            Downloaded/fine-tuned checkpoints (gitignored, not
                     redistributed -- see sections 5d/6 for how to get
                     each one)
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
    demo/         Small hand-written sample reviews for the local demo
  models/       Not committed -- downloaded/fine-tuned checkpoints, see
               sections 5d/6
  src/          Reusable, tested Python modules
    data/         XML/triplet-format parsers
    baseline/     Non-neural baselines (TF-IDF+LogReg, ASTE lookup)
    report/       Per-aspect aggregation logic, FLAN-T5 prompting/
                 generation, and the rule-based factual checker
  scripts/      Local CLIs (baseline training/eval, report generation,
               local ASTE demo)
  notebooks/    Kaggle/Colab-ready notebook templates
  notebooks-output/  Executed copies with real results
  tests/        pytest suite (58 tests, all local)
  results/      Recorded metrics JSON
  output/       Final per-aspect stats/reasons tables + generated reports
