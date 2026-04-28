"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import unicodedata
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# Core Tasks (1-6)
# ---------------------------------------------------------------------------

def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    word_counts = df["text"].str.split().str.len()

    return {
        "shape": df.shape,
        "lang_counts": df["language"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": round(word_counts.mean(), 2),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
        },
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    normalized = unicodedata.normalize("NFC", text)
    doc = nlp(normalized)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space
    ]
    return tokens


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    english_df = df[df["language"] == "en"]
    rows = []

    for _, row in english_df.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"])


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face pipeline('ner', ...) object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    english_df = df[df["language"] == "en"]
    rows = []

    for _, row in english_df.iterrows():
        results = ner_pipeline(row["text"])

        # Merge subword tokens (## continuations)
        merged = []
        for token in results:
            word = token["word"]
            if word.startswith("##") and merged:
                merged[-1]["word"] += word[2:]
                merged[-1]["end"] = token["end"]
            else:
                merged.append({
                    "word": word,
                    "entity": token["entity"],
                    "start": token["start"],
                    "end": token["end"],
                })

        for token in merged:
            # Strip IOB prefix: B-ORG -> ORG, I-ORG -> ORG
            label = token["entity"]
            if "-" in label:
                label = label.split("-", 1)[1]

            rows.append({
                "text_id": row["id"],
                "entity_text": token["word"],
                "entity_label": label,
                "start_char": token["start"],
                "end_char": token["end"],
            })

    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"])


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    spacy_set = set(zip(spacy_df["text_id"], spacy_df["entity_text"]))
    hf_set = set(zip(hf_df["text_id"], hf_df["entity_text"]))

    return {
        "spacy_counts": spacy_df["entity_label"].value_counts().to_dict(),
        "hf_counts": hf_df["entity_label"].value_counts().to_dict(),
        "total_spacy": len(spacy_df),
        "total_hf": len(hf_df),
        "both": spacy_set & hf_set,
        "spacy_only": spacy_set - hf_set,
        "hf_only": hf_set - spacy_set,
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    gold_text_ids = set(gold_df["text_id"])
    pred_filtered = predicted_df[predicted_df["text_id"].isin(gold_text_ids)]

    pred_set = set(zip(pred_filtered["text_id"], pred_filtered["entity_text"], pred_filtered["entity_label"]))
    gold_set = set(zip(gold_df["text_id"], gold_df["entity_text"], gold_df["entity_label"]))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Challenge Extension — Tier 1: Per-Category NER Analysis
# ---------------------------------------------------------------------------

def per_category_ner_analysis(entities_df, df, gold_df=None):
    """Compute entity distributions and optional evaluation per category.

    Args:
        entities_df: DataFrame of extracted entities (text_id, entity_label).
        df: Original articles DataFrame (id, category).
        gold_df: Optional gold standard DataFrame for per-category evaluation.

    Returns:
        Dictionary with keys:
          'category_label_counts': dict of category -> {label -> count}
          'category_metrics': dict of category -> {precision, recall, f1}
    """
    id_to_cat = df.set_index("id")["category"].to_dict()
    entities_df = entities_df.copy()
    entities_df["category"] = entities_df["text_id"].map(id_to_cat)

    category_label_counts = {}
    for cat, grp in entities_df.groupby("category"):
        category_label_counts[cat] = grp["entity_label"].value_counts().to_dict()

    category_metrics = {}
    if gold_df is not None:
        gold_df = gold_df.copy()
        gold_df["category"] = gold_df["text_id"].map(id_to_cat)
        for cat in gold_df["category"].dropna().unique():
            gold_cat = gold_df[gold_df["category"] == cat]
            category_metrics[cat] = evaluate_ner(entities_df, gold_cat)

    return {
        "category_label_counts": category_label_counts,
        "category_metrics": category_metrics,
    }


def plot_category_heatmap(category_label_counts, title="Entity Type by Category", save_path=None):
    """Plot a heatmap of entity type counts per category.

    Args:
        category_label_counts: dict of category -> {label -> count}
        title: Chart title string.
        save_path: Optional file path to save the figure.
    """
    categories = sorted(category_label_counts.keys())
    all_labels = sorted({
        label
        for counts in category_label_counts.values()
        for label in counts
    })

    matrix = np.array([
        [category_label_counts[cat].get(label, 0) for label in all_labels]
        for cat in categories
    ])

    fig, ax = plt.subplots(figsize=(max(10, len(all_labels)), max(4, len(categories))))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Entity Count")

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)

    for i in range(len(categories)):
        for j in range(len(all_labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Challenge Extension — Tier 2: Custom Entity Aggregation Pipeline
# ---------------------------------------------------------------------------

ENTITY_NORMALIZATION_MAP = {
    "United Nations": "UN",
    "U.N.": "UN",
    "the United Nations": "UN",
    "Intergovernmental Panel on Climate Change": "IPCC",
    "Conference of the Parties": "COP",
    "United States": "US",
    "U.S.": "US",
    "United Arab Emirates": "UAE",
    "U.A.E.": "UAE",
    "European Union": "EU",
    "E.U.": "EU",
    "World Bank Group": "World Bank",
    "WB": "World Bank",
    "Paris accord": "Paris Agreement",
}


def normalize_entities(entities_df, mapping=None):
    """Normalize entity text variants to canonical forms.

    Args:
        entities_df: DataFrame with column entity_text.
        mapping: dict of variant -> canonical.

    Returns:
        Copy of entities_df with added 'entity_canonical' column.
    """
    if mapping is None:
        mapping = ENTITY_NORMALIZATION_MAP
    df = entities_df.copy()
    df["entity_canonical"] = df["entity_text"].apply(lambda t: mapping.get(t, t))
    return df


def compute_cooccurrence(entities_df, use_canonical=True):
    """Compute entity co-occurrence counts within each text.

    Args:
        entities_df: DataFrame with columns text_id, entity_text /
                     entity_canonical.
        use_canonical: Whether to use the canonical entity name column.

    Returns:
        dict mapping (entity_a, entity_b) -> co-occurrence count.
    """
    col = "entity_canonical" if use_canonical and "entity_canonical" in entities_df.columns else "entity_text"
    cooccurrence = defaultdict(int)

    for text_id, grp in entities_df.groupby("text_id"):
        unique_entities = grp[col].unique().tolist()
        for a, b in combinations(sorted(unique_entities), 2):
            cooccurrence[(a, b)] += 1

    return dict(cooccurrence)


def compute_entity_tfidf(entities_df, df, use_canonical=True):
    """Compute TF-IDF-style entity importance per category.

    Args:
        entities_df: DataFrame with text_id and entity_text / canonical.
        df: Original articles DataFrame with id and category columns.
        use_canonical: Whether to use canonical entity name.

    Returns:
        DataFrame with columns: entity, category, tf, idf, tfidf.
    """
    col = "entity_canonical" if use_canonical and "entity_canonical" in entities_df.columns else "entity_text"
    id_to_cat = df.set_index("id")["category"].to_dict()
    entities_df = entities_df.copy()
    entities_df["category"] = entities_df["text_id"].map(id_to_cat)

    categories = entities_df["category"].dropna().unique()
    n_cats = len(categories)

    cat_entity_counts = (
        entities_df.groupby(["category", col])
        .size()
        .reset_index(name="count")
    )
    cat_totals = cat_entity_counts.groupby("category")["count"].sum().to_dict()
    entity_cat_presence = cat_entity_counts.groupby(col)["category"].nunique().to_dict()

    rows = []
    for _, row in cat_entity_counts.iterrows():
        entity = row[col]
        cat = row["category"]
        tf = row["count"] / cat_totals[cat]
        idf = np.log(n_cats / entity_cat_presence[entity])
        rows.append({"entity": entity, "category": cat, "tf": tf, "idf": idf, "tfidf": tf * idf})

    return pd.DataFrame(rows).sort_values("tfidf", ascending=False).reset_index(drop=True)


def plot_cooccurrence_network(cooccurrence, top_n=20, save_path=None):
    """Plot a co-occurrence network for the top N entity pairs.

    Args:
        cooccurrence: dict of (entity_a, entity_b) -> count.
        top_n: Number of top pairs to include.
        save_path: Optional file path to save the figure.
    """
    top_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not top_pairs:
        print("No co-occurrence data to plot.")
        return

    nodes = list({n for pair, _ in top_pairs for n in pair})
    n = len(nodes)
    angles = [2 * np.pi * i / n for i in range(n)]
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}
    max_count = max(c for _, c in top_pairs)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.axis("off")

    for (a, b), count in top_pairs:
        x_vals = [pos[a][0], pos[b][0]]
        y_vals = [pos[a][1], pos[b][1]]
        lw = 0.5 + 4.0 * count / max_count
        ax.plot(x_vals, y_vals, color="#90CAF9", linewidth=lw, alpha=0.7, zorder=1)
        mx = (x_vals[0] + x_vals[1]) / 2
        my = (y_vals[0] + y_vals[1]) / 2
        ax.text(mx, my, str(count), fontsize=6, ha="center", va="center", color="#455A64")

    for node, (x, y) in pos.items():
        ax.scatter(x, y, s=300, color="#1565C0", zorder=2)
        ax.text(x, y + 0.07, node, fontsize=7, ha="center", va="bottom",
                fontweight="bold", color="#212121")

    ax.set_title(f"Top {top_n} Entity Co-occurrences", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Co-occurrence network saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Challenge Extension — Tier 3: Custom NER Evaluator
# ---------------------------------------------------------------------------

def evaluate_ner_advanced(predicted_df, gold_df):
    """Comprehensive NER evaluation with multiple matching strategies.

    Strategies:
      - exact:         entity_text AND entity_label must match
      - partial:       overlapping character spans count as partial credit
      - type_agnostic: correct entity_text regardless of label

    Averaging:
      - micro: pool all entities, compute global TP/FP/FN
      - macro: per-text P/R/F1, averaged across texts

    Error categories:
      - true_positive:  exact match
      - boundary_error: same label, overlapping but not identical span
      - type_error:     same text, wrong label
      - spurious:       predicted entity absent from gold
      - missing:        gold entity absent from predictions

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label, start_char, end_char.
        gold_df: DataFrame with same columns.

    Returns:
        Dictionary with keys:
          'exact', 'partial', 'type_agnostic': each has 'micro' and 'macro'
          'error_distribution': counts per error category
    """
    gold_text_ids = set(gold_df["text_id"])
    pred_df = predicted_df[predicted_df["text_id"].isin(gold_text_ids)].copy()

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    def _span_overlap(ps, pe, gs, ge):
        overlap = max(0, min(pe, ge) - max(ps, gs))
        union   = max(pe, ge) - min(ps, gs)
        return overlap / union if union > 0 else 0.0

    # ── Exact ──
    pred_exact = set(zip(pred_df["text_id"], pred_df["entity_text"], pred_df["entity_label"]))
    gold_exact = set(zip(gold_df["text_id"], gold_df["entity_text"], gold_df["entity_label"]))
    exact_micro = _prf(len(pred_exact & gold_exact),
                       len(pred_exact - gold_exact),
                       len(gold_exact - pred_exact))

    per_text_exact = []
    for tid in gold_text_ids:
        p_set = {(r["entity_text"], r["entity_label"])
                 for _, r in pred_df[pred_df["text_id"] == tid].iterrows()}
        g_set = {(r["entity_text"], r["entity_label"])
                 for _, r in gold_df[gold_df["text_id"] == tid].iterrows()}
        per_text_exact.append(_prf(len(p_set & g_set), len(p_set - g_set), len(g_set - p_set)))
    exact_macro = {k: round(float(np.mean([m[k] for m in per_text_exact])), 4)
                   for k in ("precision", "recall", "f1")}

    # ── Type-agnostic ──
    pred_ta = set(zip(pred_df["text_id"], pred_df["entity_text"]))
    gold_ta = set(zip(gold_df["text_id"], gold_df["entity_text"]))
    ta_micro = _prf(len(pred_ta & gold_ta), len(pred_ta - gold_ta), len(gold_ta - pred_ta))

    per_text_ta = []
    for tid in gold_text_ids:
        p_set = set(pred_df[pred_df["text_id"] == tid]["entity_text"])
        g_set = set(gold_df[gold_df["text_id"] == tid]["entity_text"])
        per_text_ta.append(_prf(len(p_set & g_set), len(p_set - g_set), len(g_set - p_set)))
    ta_macro = {k: round(float(np.mean([m[k] for m in per_text_ta])), 4)
                for k in ("precision", "recall", "f1")}

    # ── Partial span ──
    partial_tps, partial_fps, partial_fns = 0.0, 0.0, 0.0
    per_text_partial = []

    for tid in gold_text_ids:
        p_rows = pred_df[pred_df["text_id"] == tid]
        g_rows = gold_df[gold_df["text_id"] == tid]
        tp_local = 0.0

        for _, pr in p_rows.iterrows():
            best = 0.0
            for _, gr in g_rows.iterrows():
                if pr["entity_label"] == gr["entity_label"]:
                    best = max(best, _span_overlap(
                        pr["start_char"], pr["end_char"],
                        gr["start_char"], gr["end_char"]
                    ))
            tp_local += best

        fp_local = max(len(p_rows) - tp_local, 0)
        fn_local = max(len(g_rows) - tp_local, 0)
        partial_tps += tp_local
        partial_fps += fp_local
        partial_fns += fn_local
        per_text_partial.append(_prf(tp_local, fp_local, fn_local))

    partial_micro = _prf(partial_tps, partial_fps, partial_fns)
    partial_macro = {k: round(float(np.mean([m[k] for m in per_text_partial])), 4)
                     for k in ("precision", "recall", "f1")}

    # ── Error distribution ──
    error_dist = defaultdict(int)
    for _, pr in pred_df.iterrows():
        tid = pr["text_id"]
        g_rows = gold_df[gold_df["text_id"] == tid]

        if not g_rows[(g_rows["entity_text"] == pr["entity_text"]) &
                      (g_rows["entity_label"] == pr["entity_label"])].empty:
            error_dist["true_positive"] += 1
            continue

        if not g_rows[(g_rows["entity_text"] == pr["entity_text"]) &
                      (g_rows["entity_label"] != pr["entity_label"])].empty:
            error_dist["type_error"] += 1
            continue

        found_boundary = False
        for _, gr in g_rows[g_rows["entity_label"] == pr["entity_label"]].iterrows():
            if _span_overlap(pr["start_char"], pr["end_char"],
                             gr["start_char"], gr["end_char"]) > 0:
                error_dist["boundary_error"] += 1
                found_boundary = True
                break
        if not found_boundary:
            error_dist["spurious"] += 1

    for _, gr in gold_df.iterrows():
        tid = gr["text_id"]
        p_rows = pred_df[pred_df["text_id"] == tid]
        if p_rows[(p_rows["entity_text"] == gr["entity_text"]) &
                  (p_rows["entity_label"] == gr["entity_label"])].empty:
            error_dist["missing"] += 1

    return {
        "exact":          {"micro": exact_micro,   "macro": exact_macro},
        "partial":        {"micro": partial_micro,  "macro": partial_macro},
        "type_agnostic":  {"micro": ta_micro,       "macro": ta_macro},
        "error_distribution": dict(error_dist),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nlp    = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # ── Core Tasks ──────────────────────────────────────────────────────────
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        sample_row    = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nspaCy label counts:  {comparison['spacy_counts']}")
                print(f"HF label counts:     {comparison['hf_counts']}")
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        gold = pd.read_csv("data/gold_entities.csv")

        if spacy_entities is not None:
            spacy_metrics = evaluate_ner(spacy_entities, gold)
            print(f"\nspaCy evaluation:        {spacy_metrics}")
        if hf_entities is not None:
            hf_metrics = evaluate_ner(hf_entities, gold)
            print(f"Hugging Face evaluation: {hf_metrics}")

        # ── Tier 1: Per-Category NER Analysis ───────────────────────────────
        print("\n── Tier 1: Per-Category NER Analysis ──")
        tier1 = per_category_ner_analysis(spacy_entities, df, gold_df=gold)
        for cat, counts in tier1["category_label_counts"].items():
            print(f"  {cat}: {counts}")
        if tier1["category_metrics"]:
            print("  Per-category metrics (spaCy):")
            for cat, m in tier1["category_metrics"].items():
                print(f"    {cat}: {m}")
        plot_category_heatmap(
            tier1["category_label_counts"],
            title="spaCy Entity Type by Category",
            save_path="data/category_heatmap.png",
        )

        # ── Tier 2: Entity Aggregation Pipeline ─────────────────────────────
        print("\n── Tier 2: Entity Aggregation Pipeline ──")
        spacy_norm = normalize_entities(spacy_entities)
        cooc       = compute_cooccurrence(spacy_norm)
        top10      = sorted(cooc.items(), key=lambda x: x[1], reverse=True)[:10]
        print("  Top 10 co-occurring entity pairs:")
        for (a, b), cnt in top10:
            print(f"    ({a}, {b}): {cnt}")

        tfidf_df = compute_entity_tfidf(spacy_norm, df)
        print("\n  Top 10 distinctive entities by TF-IDF:")
        print(tfidf_df.head(10).to_string(index=False))

        plot_cooccurrence_network(cooc, top_n=20, save_path="data/cooccurrence_network.png")

        # ── Tier 3: Advanced Evaluation ─────────────────────────────────────
        print("\n── Tier 3: Advanced NER Evaluation (spaCy) ──")
        adv = evaluate_ner_advanced(spacy_entities, gold)
        for strategy in ("exact", "partial", "type_agnostic"):
            print(f"  {strategy}:")
            print(f"    micro: {adv[strategy]['micro']}")
            print(f"    macro: {adv[strategy]['macro']}")
        print(f"  Error distribution: {adv['error_distribution']}")

        print("\n── Tier 3: Advanced NER Evaluation (HF) ──")
        adv_hf = evaluate_ner_advanced(hf_entities, gold)
        for strategy in ("exact", "partial", "type_agnostic"):
            print(f"  {strategy}:")
            print(f"    micro: {adv_hf[strategy]['micro']}")
            print(f"    macro: {adv_hf[strategy]['macro']}")
        print(f"  Error distribution: {adv_hf['error_distribution']}")