# 20252B-Gen-AI

Course project for Generative AI (20252B): **Aspect-Based Sentiment Analysis with Generative Report Synthesis**.

Instead of classifying a customer review as simply positive/negative, the system extracts the aspects mentioned (e.g., screen, battery, delivery, price), classifies sentiment per aspect, and uses a generative model to synthesize a short summary report across many reviews — with a lightweight factual checker to guard against the model fabricating stats/insights.

Planned dataset: [SemEval-2014 Task 4 (Laptop Reviews)](https://alt.qcri.org/semeval2014/task4/) for aspect/sentiment labels, with Amazon Reviews for large-scale report generation demos.

See [`docs/Proposal.md`](docs/Proposal.md) for the full proposal and [`docs/Requirements.md`](docs/Requirements.md) for assignment requirements.
