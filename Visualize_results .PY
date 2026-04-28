"""
visualize_results.py
====================
Generates all result visualizations for Lab 6A and Stretch 6A-S1.
Saves all plots to data/plots/.

Plots produced:
  1. bar_spacy_vs_hf_counts.png     — entity counts by label (spaCy vs HF)
  2. bar_precision_recall_f1.png    — P/R/F1 comparison (spaCy vs HF)
  3. bar_custom_rules_counts.png    — entity counts baseline vs ruler-before vs ruler-after
  4. bar_custom_labels.png          — custom label counts added by EntityRuler
  5. pie_spacy_distribution.png     — spaCy entity type distribution (pie)
  6. heatmap_overlap.png            — spaCy vs HF overlap heatmap
  7. bar_error_breakdown.png        — error breakdown TP/FP/FN/type/boundary
  8. bar_tfidf_entities.png         — top TF-IDF distinctive entities per category

Run: python visualize_results.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = "data/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_SPACY  = "#2196F3"   # blue
C_HF     = "#FF5722"   # orange-red
C_BASE   = "#9E9E9E"   # grey
C_BEFORE = "#4CAF50"   # green
C_AFTER  = "#FFC107"   # amber

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bar chart — spaCy vs HF entity counts by label
# ─────────────────────────────────────────────────────────────────────────────
def plot_spacy_vs_hf_counts():
    # spaCy labels & counts (from ner_pipeline.py output)
    spacy_labels = ["DATE","ORG","GPE","CARDINAL","PERCENT","LOC",
                    "QUANTITY","MONEY","PERSON","NORP","ORDINAL","FAC",
                    "EVENT","TIME","WORK_OF_ART","PRODUCT","LAW"]
    spacy_counts = [256,184,165,138,103,93,92,63,36,21,9,9,8,8,6,6,5]

    # HF labels (CoNLL schema — different from spaCy)
    hf_labels  = ["ORG","LOC","MISC","PER"]
    hf_counts  = [465,430,199,31]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Entity Counts by Label — spaCy vs Hugging Face", fontsize=14, fontweight="bold")

    # spaCy
    axes[0].barh(spacy_labels[::-1], spacy_counts[::-1], color=C_SPACY)
    axes[0].set_title("spaCy (en_core_web_sm)  —  1,202 total")
    axes[0].set_xlabel("Count")
    for i, v in enumerate(spacy_counts[::-1]):
        axes[0].text(v + 2, i, str(v), va="center", fontsize=8)

    # HF
    axes[1].barh(hf_labels[::-1], hf_counts[::-1], color=C_HF)
    axes[1].set_title("Hugging Face (dslim/bert-base-NER)  —  1,125 total")
    axes[1].set_xlabel("Count")
    for i, v in enumerate(hf_counts[::-1]):
        axes[1].text(v + 2, i, str(v), va="center", fontsize=8)

    fig.tight_layout()
    save(fig, "bar_spacy_vs_hf_counts.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bar chart — Precision / Recall / F1
# ─────────────────────────────────────────────────────────────────────────────
def plot_precision_recall_f1():
    systems = ["spaCy", "HF"]
    precision = [0.6286, 0.0787]
    recall    = [0.6471, 0.1029]
    f1        = [0.6377, 0.0892]

    x = np.arange(len(systems))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, precision, w, label="Precision", color="#42A5F5")
    ax.bar(x,     recall,    w, label="Recall",    color="#66BB6A")
    ax.bar(x + w, f1,        w, label="F1",        color="#FFA726")

    ax.set_title("Precision / Recall / F1 — Gold Standard Subset (10 texts)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score")
    ax.legend()
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    fig.tight_layout()
    save(fig, "bar_precision_recall_f1.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bar chart — entity counts baseline vs ruler-before vs ruler-after
# ─────────────────────────────────────────────────────────────────────────────
def plot_custom_rules_counts():
    systems = ["Baseline\n(no rules)", "Ruler BEFORE NER", "Ruler AFTER NER"]
    totals  = [1202, 1215, 1213]
    colors  = [C_BASE, C_BEFORE, C_AFTER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(systems, totals, color=colors, width=0.5, edgecolor="white")
    ax.set_title("Total Entity Counts — Custom Rules Impact", fontweight="bold")
    ax.set_ylabel("Total Entities Extracted")
    ax.set_ylim(1180, 1230)
    ax.bar_label(bars, padding=3, fontsize=11, fontweight="bold")

    # delta annotations
    ax.annotate("+13 vs baseline", xy=(1, 1215), xytext=(1, 1222),
                ha="center", fontsize=9, color=C_BEFORE,
                arrowprops=dict(arrowstyle="->", color=C_BEFORE))
    ax.annotate("+11 vs baseline", xy=(2, 1213), xytext=(2, 1225),
                ha="center", fontsize=9, color=C_AFTER,
                arrowprops=dict(arrowstyle="->", color=C_AFTER))

    fig.tight_layout()
    save(fig, "bar_custom_rules_counts.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bar chart — custom label counts (ruler BEFORE only)
# ─────────────────────────────────────────────────────────────────────────────
def plot_custom_labels():
    labels = ["CLIMATE_EVENT", "POLICY", "THRESHOLD", "REPORT"]
    counts = [10, 15, 8, 2]
    colors = ["#7E57C2", "#26A69A", "#EF5350", "#8D6E63"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white")
    ax.set_title("Custom Entity Labels Added by EntityRuler (Ruler BEFORE NER)", fontweight="bold")
    ax.set_ylabel("Entities Found")
    ax.bar_label(bars, padding=3, fontsize=12, fontweight="bold")

    examples = {
        "CLIMATE_EVENT": "COP28, Bonn Conference\nClimate Ambition Summit",
        "POLICY":        "Paris Agreement\nCarbon Border Adj. Mechanism, NDC",
        "THRESHOLD":     "1.5 degrees Celsius\n2 degrees Celsius",
        "REPORT":        "Sixth Assessment Report\nState of Food & Agriculture",
    }
    for i, (label, ex) in enumerate(examples.items()):
        ax.text(i, -1.5, ex, ha="center", va="top", fontsize=7,
                color="#555", style="italic")

    ax.set_ylim(-5, 20)
    fig.tight_layout()
    save(fig, "bar_custom_labels.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pie chart — spaCy entity type distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_pie_spacy():
    labels = ["DATE","ORG","GPE","CARDINAL","PERCENT","LOC",
              "QUANTITY","MONEY","PERSON","Other"]
    counts = [256,184,165,138,103,93,92,63,36,
              9+9+8+8+6+6+5]   # NORP+ORDINAL+FAC+EVENT+TIME+WORK_OF_ART+PRODUCT+LAW

    colors = ["#EF5350","#42A5F5","#66BB6A","#FFA726","#AB47BC",
              "#26C6DA","#D4E157","#FF7043","#78909C","#BDBDBD"]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=140,
        pctdistance=0.82, wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(8)

    ax.legend(wedges, [f"{l} ({c})" for l, c in zip(labels, counts)],
              loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_title("spaCy Entity Type Distribution (1,202 entities)", fontweight="bold")
    fig.tight_layout()
    save(fig, "pie_spacy_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Heatmap — spaCy vs HF overlap
# ─────────────────────────────────────────────────────────────────────────────
def plot_overlap_heatmap():
    # From output: agreed=230, spacy_only=958, hf_only=826
    # Total unique = 230+958+826 = 2014
    total = 230 + 958 + 826

    data = np.array([
        [230,  958],   # HF found: both | spaCy only
        [826,    0],   # HF only  | neither (N/A)
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow([[230, 958],[826, 0]], cmap="Blues", vmin=0, vmax=1000)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Found by HF", "Not found by HF"], fontsize=11)
    ax.set_yticklabels(["Found by spaCy", "Not found by spaCy"], fontsize=11)
    ax.set_title("spaCy vs HF Entity Overlap\n(matched on text_id + entity_text)",
                 fontweight="bold")

    labels_map = {
        (0,0): f"Both\n230",
        (0,1): f"spaCy only\n958",
        (1,0): f"HF only\n826",
        (1,1): "—",
    }
    for (r, c), txt in labels_map.items():
        ax.text(c, r, txt, ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if (r==0 and c==1) else "#333")

    fig.colorbar(im, ax=ax, label="Entity Count")
    fig.tight_layout()
    save(fig, "heatmap_overlap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bar chart — error breakdown (spaCy vs HF)
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_breakdown():
    categories = ["True Positive", "Spurious\n(FP)", "Missing\n(FN)",
                  "Boundary\nError", "Type\nError"]
    spacy_vals = [45, 13, 24, 8, 5]
    hf_vals    = [8,  57, 61, 19, 8]

    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, spacy_vals, w, label="spaCy", color=C_SPACY)
    b2 = ax.bar(x + w/2, hf_vals,    w, label="HF",    color=C_HF)

    ax.set_title("Error Breakdown — spaCy vs HF (Gold Standard Subset)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Count")
    ax.legend(fontsize=11)
    ax.bar_label(b1, padding=3, fontsize=9)
    ax.bar_label(b2, padding=3, fontsize=9)

    fig.tight_layout()
    save(fig, "bar_error_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Bar chart — TF-IDF top distinctive entities per category
# ─────────────────────────────────────────────────────────────────────────────
def plot_tfidf_entities():
    # Top entities from output (entity, category, tfidf)
    data = [
        ("the Paris Agreement", "policy",     0.025023),
        ("$1.3 trillion",       "policy",     0.015014),
        ("Red Sea",             "science",    0.013503),
        ("Arctic",              "science",    0.013503),
        ("NASA",                "science",    0.013503),
        ("Morocco",             "adaptation", 0.011951),
        ("Europe",              "adaptation", 0.011951),
        ("the Middle East",     "science",    0.011252),
        ("Turkey",              "impact",     0.010307),
        ("2022",                "policy",     0.012512),
    ]
    # Sort by tfidf
    data.sort(key=lambda x: x[2], reverse=True)

    entities   = [d[0] for d in data]
    tfidf_vals = [d[2] for d in data]
    cat_colors = {
        "policy":     "#42A5F5",
        "science":    "#66BB6A",
        "adaptation": "#FFA726",
        "impact":     "#EF5350",
    }
    colors = [cat_colors[d[1]] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(entities[::-1], tfidf_vals[::-1], color=colors[::-1])
    ax.set_title("Top Distinctive Entities by TF-IDF Score (per Category)", fontweight="bold")
    ax.set_xlabel("TF-IDF Score")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)

    # legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    fig.tight_layout()
    save(fig, "bar_tfidf_entities.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating visualizations...")
    plot_spacy_vs_hf_counts()
    plot_precision_recall_f1()
    plot_custom_rules_counts()
    plot_custom_labels()
    plot_pie_spacy()
    plot_overlap_heatmap()
    plot_error_breakdown()
    plot_tfidf_entities()
    print(f"\nDone — all plots saved to {OUT_DIR}/")