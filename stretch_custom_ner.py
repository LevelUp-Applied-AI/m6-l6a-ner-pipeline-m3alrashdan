"""
Module 6 Week A — Stretch: Custom NER Rules

Extends the base spaCy NER pipeline with a domain-specific EntityRuler
for climate terminology. Tests the ruler in two pipeline positions
(before and after the statistical NER) and evaluates the impact on
entity extraction quality against the gold standard.

Run: python stretch_custom_ner.py
"""

import unicodedata
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

# Gold-standard labels used for evaluation
# Custom labels (CLIMATE_EVENT, POLICY, REPORT, THRESHOLD) are evaluated qualitatively
STANDARD_LABELS = {
    "ORG", "GPE", "DATE", "LAW", "MONEY", "PERSON",
    "QUANTITY", "LOC", "EVENT", "WORK_OF_ART"
}


# ---------------------------------------------------------------------------
# Custom EntityRuler patterns
# ---------------------------------------------------------------------------

CLIMATE_PATTERNS = [
    # ── CLIMATE_EVENT: named climate conferences and summits ──────────────
    # COP26 / COP27 / COP28  (token pattern handles any COP+number)
    {"label": "CLIMATE_EVENT", "pattern": [{"TEXT": {"REGEX": r"^COP\d{1,2}$"}}]},
    # Bonn Climate Change Conference (phrase pattern)
    {"label": "CLIMATE_EVENT", "pattern": [{"LOWER": "bonn"}, {"LOWER": "climate"},
                                            {"LOWER": "change"}, {"LOWER": "conference"}]},
    # Climate Ambition Summit
    {"label": "CLIMATE_EVENT", "pattern": [{"LOWER": "climate"}, {"LOWER": "ambition"},
                                            {"LOWER": "summit"}]},
    # Global Stocktake
    {"label": "CLIMATE_EVENT", "pattern": [{"LOWER": "global"}, {"LOWER": "stocktake"}]},

    # ── POLICY: international agreements and legal instruments ────────────
    # Paris Agreement
    {"label": "POLICY", "pattern": [{"LOWER": "paris"}, {"LOWER": "agreement"}]},
    # Kyoto Protocol
    {"label": "POLICY", "pattern": [{"LOWER": "kyoto"}, {"LOWER": "protocol"}]},
    # Carbon Border Adjustment Mechanism
    {"label": "POLICY", "pattern": [{"LOWER": "carbon"}, {"LOWER": "border"},
                                     {"LOWER": "adjustment"}, {"LOWER": "mechanism"}]},
    # NDC / Nationally Determined Contribution
    {"label": "POLICY", "pattern": [{"TEXT": "NDC"}]},
    {"label": "POLICY", "pattern": [{"LOWER": "nationally"}, {"LOWER": "determined"},
                                     {"LOWER": "contribution"}]},
    # Net Zero / Net-Zero target
    {"label": "POLICY", "pattern": [{"LOWER": {"IN": ["net-zero", "net"]}},
                                     {"LOWER": "zero", "OP": "?"}]},

    # ── REPORT: major climate science and policy reports ──────────────────
    # Sixth Assessment Report (and variants)
    {"label": "REPORT", "pattern": [{"LOWER": {"IN": ["sixth", "fifth", "fourth"]}},
                                     {"LOWER": "assessment"}, {"LOWER": "report"}]},
    # State of Food and Agriculture
    {"label": "REPORT", "pattern": [{"LOWER": "state"}, {"LOWER": "of"},
                                     {"LOWER": "food"}, {"LOWER": "and"},
                                     {"LOWER": "agriculture"}]},

    # ── THRESHOLD: scientific temperature and emissions limits ────────────
    # 1.5 degrees Celsius
    {"label": "THRESHOLD", "pattern": [{"TEXT": "1.5"}, {"LOWER": {"IN": ["degrees", "degree"]}},
                                        {"LOWER": "celsius"}]},
    # 2 degrees Celsius
    {"label": "THRESHOLD", "pattern": [{"TEXT": "2"}, {"LOWER": {"IN": ["degrees", "degree"]}},
                                        {"LOWER": "celsius"}]},
    # 1.5°C shorthand
    {"label": "THRESHOLD", "pattern": [{"TEXT": {"REGEX": r"^1\.5[°℃]?C?$"}}]},
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data(filepath="data/climate_articles.csv"):
    return pd.read_csv(filepath)


def load_gold(filepath="data/gold_entities.csv"):
    return pd.read_csv(filepath)


def build_nlp_with_ruler(position="before"):
    """
    Load en_core_web_sm and insert the EntityRuler at the specified position.

    position='before'  → ruler runs first; its matches override the stat model
    position='after'   → stat model runs first; ruler fills gaps
    """
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner" if position == "before" else None,
        config={"overwrite_ents": position == "before"},
    )
    ruler.add_patterns(CLIMATE_PATTERNS)
    return nlp


def extract_entities(df, nlp):
    """Extract entities from English texts using the given pipeline."""
    english_df = df[df["language"] == "en"].copy()
    rows = []
    for _, row in english_df.iterrows():
        doc = nlp(str(row["text"]))
        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label",
                                        "start_char", "end_char"])


def evaluate_standard_labels(predicted_df, gold_df):
    """
    Evaluate only on entities whose label appears in STANDARD_LABELS.
    Custom labels (CLIMATE_EVENT, POLICY, REPORT, THRESHOLD) are excluded
    because the gold standard has no entries for them — including them would
    artificially depress precision.
    """
    pred_filtered = predicted_df[predicted_df["entity_label"].isin(STANDARD_LABELS)]

    pred_set = set(zip(pred_filtered["text_id"],
                       pred_filtered["entity_text"],
                       pred_filtered["entity_label"]))
    gold_set = set(zip(gold_df["text_id"],
                       gold_df["entity_text"],
                       gold_df["entity_label"]))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def print_label_counts(entities_df, label=""):
    counts = entities_df["entity_label"].value_counts()
    print(f"\n{'Label':<20} {'Count':>6}   [{label}]")
    print("-" * 35)
    for lbl, cnt in counts.items():
        print(f"  {lbl:<18} {cnt:>6}")
    print(f"  {'TOTAL':<18} {len(entities_df):>6}")


def show_custom_rule_examples(entities_df, df, n=3):
    """Print sample texts where each custom label fired."""
    custom_labels = ["CLIMATE_EVENT", "POLICY", "REPORT", "THRESHOLD"]
    for label in custom_labels:
        hits = entities_df[entities_df["entity_label"] == label]
        if hits.empty:
            print(f"\n  {label}: no matches found")
            continue
        print(f"\n  {label} ({len(hits)} matches):")
        for _, hit in hits.drop_duplicates("entity_text").head(n).iterrows():
            src = df[df["id"] == hit["text_id"]]["text"].values[0]
            snippet = src[max(0, hit["start_char"]-30):hit["end_char"]+30].replace("\n", " ")
            print(f"    • \"{hit['entity_text']}\"  → ...{snippet}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df   = load_data()
    gold = load_gold()

    # ── Baseline: plain en_core_web_sm, no custom rules ──────────────────
    print("=" * 60)
    print("BASELINE — en_core_web_sm (no custom rules)")
    print("=" * 60)
    nlp_base     = spacy.load("en_core_web_sm")
    base_ents    = extract_entities(df, nlp_base)
    base_metrics = evaluate_standard_labels(base_ents, gold)
    print_label_counts(base_ents, "baseline")
    print(f"\n  Precision: {base_metrics['precision']}  "
          f"Recall: {base_metrics['recall']}  "
          f"F1: {base_metrics['f1']}")
    print(f"  TP={base_metrics['tp']}  FP={base_metrics['fp']}  FN={base_metrics['fn']}")

    # ── Position A: EntityRuler BEFORE statistical NER ────────────────────
    print("\n" + "=" * 60)
    print("POSITION A — EntityRuler BEFORE NER (ruler takes priority)")
    print("=" * 60)
    nlp_before     = build_nlp_with_ruler(position="before")
    before_ents    = extract_entities(df, nlp_before)
    before_metrics = evaluate_standard_labels(before_ents, gold)
    print_label_counts(before_ents, "ruler before NER")
    print(f"\n  Precision: {before_metrics['precision']}  "
          f"Recall: {before_metrics['recall']}  "
          f"F1: {before_metrics['f1']}")
    print(f"  TP={before_metrics['tp']}  FP={before_metrics['fp']}  FN={before_metrics['fn']}")

    print("\n  Custom rule examples (ruler BEFORE):")
    show_custom_rule_examples(before_ents, df)

    # ── Position B: EntityRuler AFTER statistical NER ─────────────────────
    print("\n" + "=" * 60)
    print("POSITION B — EntityRuler AFTER NER (stat model takes priority)")
    print("=" * 60)
    nlp_after     = build_nlp_with_ruler(position="after")
    after_ents    = extract_entities(df, nlp_after)
    after_metrics = evaluate_standard_labels(after_ents, gold)
    print_label_counts(after_ents, "ruler after NER")
    print(f"\n  Precision: {after_metrics['precision']}  "
          f"Recall: {after_metrics['recall']}  "
          f"F1: {after_metrics['f1']}")
    print(f"  TP={after_metrics['tp']}  FP={after_metrics['fp']}  FN={after_metrics['fn']}")

    print("\n  Custom rule examples (ruler AFTER):")
    show_custom_rule_examples(after_ents, df)

    # ── Summary delta table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — Before / After Custom Rules")
    print("=" * 60)
    print(f"\n{'System':<30} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for name, m in [("Baseline (no rules)",   base_metrics),
                    ("Ruler BEFORE NER",       before_metrics),
                    ("Ruler AFTER NER",        after_metrics)]:
        print(f"  {name:<28} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    print(f"\n  Precision delta (before): "
          f"{before_metrics['precision'] - base_metrics['precision']:+.4f}")
    print(f"  Recall    delta (before): "
          f"{before_metrics['recall'] - base_metrics['recall']:+.4f}")
    print(f"  F1        delta (before): "
          f"{before_metrics['f1'] - base_metrics['f1']:+.4f}")