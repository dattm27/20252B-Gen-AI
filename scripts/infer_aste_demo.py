"""CLI: run a fine-tuned T5 ASTE model (see notebooks/train-t5-base-for-aste-on-14res-15res-16res.ipynb)
over a small local sample of restaurant-domain sentences, entirely on CPU — for a live demo, not
full-dataset evaluation (that stays on Kaggle GPU, see README "Aspect + top-10-reasons (t5-base)").

The fine-tuned checkpoint isn't in the repo (gitignored, trained on Kaggle) — download it from the
training notebook's Output tab and point --model at the extracted folder
(default: models/t5-base-aste-restaurant-best).

Usage:
    python scripts/infer_aste_demo.py \
        --model models/t5-base-aste-restaurant-best \
        --input data/demo/sample_reviews.json \
        --output output/aspect_reasons_demo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.aste_loader import text_to_triplets
from src.report.aspect_stats import aggregate_aspect_reasons

PREFIX = "extract aspect sentiment triplets: "


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/t5-base-aste-restaurant-best")
    parser.add_argument("--input", default="data/demo/sample_reviews.json")
    parser.add_argument("--output", default="output/aspect_reasons_demo.json")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=160)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--min-mentions", type=int, default=1)
    parser.add_argument("--top-n-reasons", type=int, default=10)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    samples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(f"Loaded {len(samples)} sample sentences from {args.input}")

    print(f"Loading model from {args.model} (CPU) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()

    predictions = []
    records = []
    for i, sample in enumerate(samples, 1):
        text = sample["text"]
        inputs = tokenizer(
            PREFIX + text, return_tensors="pt", truncation=True, max_length=args.max_input_length
        )
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=args.max_target_length,
                num_beams=args.num_beams,
                early_stopping=True,
            )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        triplets = text_to_triplets(generated)

        print(f"[{i}/{len(samples)}] ({sample.get('case', '?')}) {text}")
        print(f"    -> {generated}")

        predictions.append({
            "case": sample.get("case"),
            "text": text,
            "generated": generated,
            "triplets": [{"aspect": t.aspect, "opinion": t.opinion, "sentiment": t.sentiment} for t in triplets],
        })
        records.extend((t.aspect, t.opinion, t.sentiment) for t in triplets)

    summaries = aggregate_aspect_reasons(records, top_n=args.top_n_reasons, min_mentions=args.min_mentions)

    result = {
        "model_dir": args.model,
        "domain": "restaurant (local demo sample)",
        "min_mentions": args.min_mentions,
        "top_n_reasons": args.top_n_reasons,
        "num_examples": len(samples),
        "predictions": predictions,
        "predicted": [
            {
                "aspect": s.aspect,
                "positive": s.positive,
                "positive_reasons": s.positive_reasons,
                "negative": s.negative,
                "negative_reasons": s.negative_reasons,
                "neutral": s.neutral,
                "neutral_reasons": s.neutral_reasons,
                "total": s.total,
                "majority_sentiment": s.majority_sentiment,
            }
            for s in summaries
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{len(summaries)} aspects aggregated. Saved to {out_path}")


if __name__ == "__main__":
    main()
