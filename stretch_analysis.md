# Stretch 6A-S1 — Custom NER Rules: Analysis

## Custom Entity Types and Pattern Design

Four custom labels were added via `EntityRuler`, covering 15 pattern entries across 12 distinct concepts:

| Label             | Concepts Covered                                                                         | Example Match                                                     |
| ----------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `CLIMATE_EVENT` | COP26/27/28, Bonn Conference, Climate Ambition Summit, Global Stocktake                  | "At**COP28** in Dubai..."                                   |
| `POLICY`        | Paris Agreement, Kyoto Protocol, CBAM, NDC, Nationally Determined Contribution, Net Zero | "fall short of the**Paris Agreement** targets"              |
| `REPORT`        | Sixth/Fifth/Fourth Assessment Report, State of Food and Agriculture                      | "released its**Sixth Assessment Report**"                   |
| `THRESHOLD`     | 1.5 degrees Celsius, 2 degrees Celsius, 1.5°C shorthand                                 | "exceed**1.5 degrees Celsius** above pre-industrial levels" |

---

## Before / After Metrics (standard labels only)

Evaluation is restricted to the 10 gold-standard labels (`ORG`, `GPE`, `DATE`, `LAW`, `MONEY`, `PERSON`, `QUANTITY`, `LOC`, `EVENT`, `WORK_OF_ART`). Custom labels are excluded because the gold standard has no entries for `CLIMATE_EVENT`, `POLICY`, `REPORT`, or `THRESHOLD` — including them would artificially depress precision.

| System              | Precision | Recall | F1     | TP | FP  | FN |
| ------------------- | --------- | ------ | ------ | -- | --- | -- |
| Baseline (no rules) | 0.0492    | 0.6471 | 0.0914 | 44 | 851 | 24 |
| Ruler BEFORE NER    | 0.0514    | 0.6618 | 0.0954 | 45 | 830 | 23 |
| Ruler AFTER NER     | 0.0492    | 0.6471 | 0.0914 | 44 | 851 | 24 |

**Delta (ruler BEFORE vs baseline):** Precision +0.0022 · Recall +0.0147 · F1 +0.0040

> Note on the low precision: the gold standard covers only 10 of 132 English texts (~69 entities). The model predicts entities across all 132 texts, so most predictions land outside the annotated set and count as false positives by construction — this is expected given the small gold subset, not a sign of a broken model.

---

## Pipeline Position: Before vs. After

**Ruler BEFORE NER** (`overwrite_ents=True`) produced 1,215 total entities — 13 more than baseline — including 10 `CLIMATE_EVENT`, 15 `POLICY`, 8 `THRESHOLD`, and 2 `REPORT` matches. The ruler fires first and claims its spans; the statistical NER then skips those tokens. This gave a small but real improvement: +1 TP, -21 FP, -1 FN. The best gains came from `CLIMATE_EVENT` (capturing `COP28`, `Bonn Climate Change Conference`, `Climate Ambition Summit`) and `THRESHOLD` (capturing `1.5 degrees Celsius`, `2 degrees Celsius`), which the base model either missed or labelled as generic `QUANTITY`.

**Ruler AFTER NER** produced identical metrics to the baseline (Precision 0.0492, Recall 0.6471, F1 0.0914). The statistical model had already claimed most relevant spans, leaving little room for the ruler to add new entities. Only 5 `CLIMATE_EVENT` and 6 `POLICY` matches survived — and `REPORT` and `THRESHOLD` found zero matches because the base model's `WORK_OF_ART` and `QUANTITY` spans covered those tokens first. This confirms that **ruler-before is the better position** for this domain, where the custom labels cover concepts the stat model was not trained on.

---

## Where Custom Rules Helped

**`CLIMATE_EVENT` — COP variants (ruler before):**
The base model inconsistently labels `COP28` or ignores it. The token regex `COP\d{1,2}` reliably captured all three variants across the corpus. In text ID 2: *"At **COP28** in Dubai, over 190 nations agreed to transition away from fossil fuels."*

**`POLICY` — Paris Agreement:**
In text ID 5, the base model tagged "Paris" as `GPE`, losing "Agreement" and misrepresenting the entity as a city rather than a legal instrument. The phrase pattern correctly captured the full span: *"fall short of the **Paris Agreement** targets."*

**`REPORT` — Sixth Assessment Report:**
The base model labelled "Sixth Assessment Report" as `WORK_OF_ART` in text ID 1, which is defensible but loses the scientific report semantics. The custom `REPORT` label captures it with the correct domain meaning: *"The IPCC released its **Sixth Assessment Report** in March 2023."*

**`THRESHOLD` — Temperature limits:**
"1.5 degrees Celsius" was tagged `QUANTITY` by the base model — technically correct but undifferentiated from any other measurement. The custom label distinguishes climate policy thresholds from arbitrary quantities: *"global temperatures could exceed **1.5 degrees Celsius** above pre-industrial levels."*

---

## Where Custom Rules Introduced Noise

**`POLICY` — Net Zero false positive:**
The `net-zero` / `net zero` pattern matched the word `"net"` alone in the text *"Decarbonization Plan targets net-zero emissions by 2050"* — the hyphenated token was split, leaving `"net"` as a spurious `POLICY` entity. This was the most visible false positive across both pipeline positions (appeared in both ruler-before and ruler-after). The fix is to replace the current pattern with a stricter two-token pattern requiring both `net` and `zero` to be present as adjacent tokens, or use a phrase pattern `"net-zero"` that matches the hyphenated form exactly.

**`CLIMATE_EVENT` — Position sensitivity:**
In ruler-after mode, only 5 of the 10 `CLIMATE_EVENT` matches survived because the base model had already claimed spans like `"Bonn Climate Change Conference"` under `ORG`. This is not a false positive, but it shows that the ruler-after position silently drops correct custom matches without any warning.

---

## Summary

Custom rules meaningfully extended the base model for climate-specific terminology — particularly for COP event names, international agreements, and temperature thresholds the general-purpose model was never trained to label. The ruler-before position was strictly better: +1 TP, -21 FP, -1 FN compared to baseline, with 37 new custom-label entities added across the corpus. The main noise source was the `net-zero` pattern, which matched a single token due to spaCy's hyphen tokenisation. In a production system, phrase patterns (`"net-zero"` as a single string) are safer than multi-token rules for hyphenated terms.
