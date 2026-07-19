"""Parser for SemEval-2014 Task 4 ABSA XML files (aspect term subtask)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AspectTerm:
    term: str
    polarity: str
    start: int
    end: int


@dataclass
class Sentence:
    sentence_id: str
    text: str
    aspect_terms: list[AspectTerm] = field(default_factory=list)


def load_semeval_xml(path: str | Path) -> list[Sentence]:
    """Load a SemEval-2014 Task 4 XML file (e.g. Laptop_Train_v2.xml).

    Sentences with no `polarity` attribute (unlabeled phase-A test data) are
    skipped at the aspect-term level but the sentence itself is still returned.
    """
    root = ET.parse(path).getroot()
    sentences = []
    for sent_el in root.findall("sentence"):
        text = sent_el.findtext("text") or ""
        aspect_terms = []
        for term_el in sent_el.findall("./aspectTerms/aspectTerm"):
            polarity = term_el.get("polarity")
            if polarity is None:
                continue
            aspect_terms.append(
                AspectTerm(
                    term=term_el.get("term", ""),
                    polarity=polarity,
                    start=int(term_el.get("from", -1)),
                    end=int(term_el.get("to", -1)),
                )
            )
        sentences.append(
            Sentence(sentence_id=sent_el.get("id", ""), text=text, aspect_terms=aspect_terms)
        )
    return sentences
