# Task B error analysis — per-species F1 and confusion pairs

**Source of every number below:** `docs/reports/experiments/phase_1b/final_b_new_recomputed/20260819T043733Z/predictions.csv`
(`split=test`, `experiment=main`) and `per_class_metrics.csv` from the same registry.
Registry created 2026-08-19T04:37:33Z, split version `task-B-group-split:359be7f5379818d4`.

Scope: 5 backbones × 3 seeds (42, 123, 456) × 532 test images = **7,980 predictions, 204 errors**.

Dispersion throughout is the standard deviation across the three seeds. It is not
an interval estimate, and no claim of statistical reliability is attached to any
difference reported here.

---

## 1. Error budget by architecture

| Backbone | Errors / 1,596 | Error rate |
|---|---|---|
| ResNet-50 | 122 | 7.64% |
| EfficientNet-B3 | 35 | 2.19% |
| MobileNetV3-Large | 26 | 1.63% |
| Swin-Tiny | 14 | 0.88% |
| ConvNeXt-Tiny | 7 | 0.44% |

ResNet-50 alone accounts for **122 of the 204 errors (60%)**. Any statement about
"which species are hard" that pools all five backbones is therefore largely a
statement about ResNet-50, and is reported separately below for that reason.

---

## 2. *Pinus krempfii* → *Pinus dalatensis* — the one architecture-independent confusion

This is the **only confusion pair that appears in every backbone**:

| Backbone | krempfii → dalatensis | dalatensis → krempfii |
|---|---|---|
| ResNet-50 | 18 | 1 |
| EfficientNet-B3 | 12 | 0 |
| MobileNetV3-Large | 15 | 0 |
| Swin-Tiny | 9 | 0 |
| ConvNeXt-Tiny | 7 | 0 |
| **Total** | **61** | **1** |

Three properties make this the substantive finding of the error analysis:

**It survives architecture.** 61 of 204 errors (30%) fall on this single pair, and it
persists even in the strongest model. For ConvNeXt-Tiny it is not merely the largest
error — it is the *only* error: all 7 of its misclassifications across all three
seeds are this pair, and its confusion matrix is otherwise exactly diagonal. A
difficulty that does not diminish as capacity grows from ResNet-50 to ConvNeXt-Tiny
is better read as a property of the image data than of any one model.

**It is strongly asymmetric.** 61 errors in the krempfii → dalatensis direction
against 1 in the reverse. *P. krempfii* images are absorbed into *P. dalatensis*;
*P. dalatensis* images are almost never mistaken for *P. krempfii*. A symmetric
"these two look alike" account does not fit — the asymmetry points to *P. krempfii*
occupying a region of feature space that overlaps *P. dalatensis*, while
*P. dalatensis* retains distinguishing evidence of its own. Support is comparable
(36 vs 32 test images), so this is not an artefact of class imbalance.

**It is within-genus and within-family.** Both are *Pinus* (Pinaceae), so the
confusion never crosses a family boundary. Consistent with this, both species
carry the lowest per-species F1 of the whole set for the two best models:
ConvNeXt-Tiny 96.573 ± 3.495 (krempfii) and 96.559 ± 3.295 (dalatensis);
Swin-Tiny 95.548 ± 4.082 and 95.620 ± 3.706. Note also that in both of
those columns these two species carry the largest seed-to-seed standard deviation
of any species — 3.30 to 4.08 points. For ConvNeXt-Tiny they are the *only* two
species with a non-zero spread at all; for Swin-Tiny the next largest are
*Podocarpus pilgeri* (± 1.70) and *Chamaecyparis obtusa* (± 1.65), less than half
as large. The size of this effect is therefore itself unstable across seeds.
(Larger spreads occur elsewhere in the table, but only for ResNet-50; see §3.)

**What this analysis does not establish.** No wood-anatomical claim is made here.
Confirming that these two species share diagnostic cross-section features would
require inspecting the images themselves (Grad-CAM over the misclassified set, and
review by someone with wood-anatomy expertise), which is outside what
`predictions.csv` can support.

---

## 3. *Podocarpus pilgeri* → *Chamaecyparis obtusa* — a ResNet-50 artefact, not a species property

| Backbone | Errors on this pair |
|---|---|
| ResNet-50 | 48 |
| Swin-Tiny | 5 |
| EfficientNet-B3 | 2 |
| MobileNetV3-Large | 0 |
| ConvNeXt-Tiny | 0 |
| **Total** | **55** |

**48 of the 55 errors (87%) come from ResNet-50 alone**, and two of the five
backbones never make this mistake at all. The pooled count of 55 makes the pair look
like the second-largest difficulty in the dataset; disaggregated by backbone it
is almost entirely one model's failure mode.

The per-species F1 for *Podocarpus pilgeri* makes the same point:

| ResNet-50 | EfficientNet-B3 | MobileNetV3-L | ConvNeXt-Tiny | Swin-Tiny |
|---|---|---|---|---|
| 61.994 ± 9.598 | 95.420 ± 1.562 | 100.000 ± 0.000 | 100.000 ± 0.000 | 97.612 ± 1.697 |

ResNet-50 sits 33.4 points below the next-worst model (EfficientNet-B3, 95.420) and carries by far the
largest seed-to-seed spread of any cell in the table (± 9.6). Two architectures
score a perfect 100.

**This pair must not be given a biological interpretation.** *Podocarpus pilgeri*
(Podocarpaceae) and *Chamaecyparis obtusa* (Cupressaceae) are in different families,
so a genuine shared-anatomy explanation would be a strong claim — and the evidence
contradicts it, because four of five architectures identify *P. pilgeri* at 95–100%
F1 on the same images. The correct reading is that ResNet-50 fails to learn a
separable representation for this species under this training configuration, not
that the two woods resemble each other.

The same caution applies to the other ResNet-heavy pairs: *Taxus wallichiana* →
*Podocarpus pilgeri* (12 of 17 from ResNet-50) and *Araucaria araucana* →
*Pinus radiata* (7 of 7 from ResNet-50, zero elsewhere).

---

## 4. Family-level difficulty — Cupressaceae is not the hardest family

Family assignment follows the **five-family** scheme: *Araucaria araucana* is
placed in Araucariaceae rather than folded into Pinaceae. Error rate by true
family, over all predictions in scope:

| Family | Species | All 5 backbones | Excluding ResNet-50 |
|---|---|---|---|
| Araucariaceae | 1 | 7 / 450 = 1.56% | 0 / 360 = **0.00%** |
| Cupressaceae | 8 | 30 / 3,840 = **0.78%** | 15 / 3,072 = 0.49% |
| Pinaceae | 6 | 78 / 2,760 = 2.83% | 47 / 2,208 = 2.13% |
| Podocarpaceae | 1 | 59 / 540 = 10.93% | 9 / 432 = 2.08% |
| Taxaceae | 1 | 30 / 390 = 7.69% | 11 / 312 = 3.53% |

**Cupressaceae has the lowest error rate over all five backbones, and any prior
expectation that it is the hardest group is not supported by this data.** All
eight of its species are error-free for ConvNeXt-Tiny. Note however that the
claim does not hold under both groupings: once ResNet-50 is excluded,
Araucariaceae drops to 0.00% and takes the lowest position, with Cupressaceae
second at 0.49%.

Three cautions on reading this table:

- **Three of the five families contain exactly one species** (*A. araucana*,
  *P. pilgeri*, *T. wallichiana*), so their family rate is a single-species rate
  over a much smaller denominator — not comparable in kind to the 8-species
  Cupressaceae and 6-species Pinaceae rates.
- **Podocarpaceae's apparent difficulty is the ResNet-50 effect from §3**: 10.93%
  falls to 2.08% once ResNet-50 is excluded, a five-fold drop. Taxaceae behaves
  similarly (7.69% → 3.53%), and Araucariaceae goes to zero — all 7 of its errors
  come from ResNet-50, every one of them *A. araucana* → *Pinus radiata*.
- **The ordering is not stable between the two groupings.** Araucariaceae moves
  from third to first and Pinaceae from third-worst to second-worst once
  ResNet-50 is removed, so no family ranking beyond "Cupressaceae is not the
  hardest" should be carried into the write-up.

Of the 204 errors, 117 cross a family boundary and 87 stay within one — but the
cross-family total is itself dominated by the ResNet-50 *P. pilgeri* failures,
and under the five-family scheme the 7 *A. araucana* → *P. radiata* errors count
as cross-family as well.

---

## 5. Traceability

| Output | Built from |
|---|---|
| `fig_perclass_f1.pdf` / `.png` | `per_class_metrics.csv` |
| `perclass_f1_summary.csv` | `per_class_metrics.csv` |
| `perclass_metadata.json` | `per_class_metrics.csv` + `predictions.csv` |
| §2–§4 of this document | `predictions.csv` (recomputed directly, not read from any summary) |

`perclass_metadata.json` carries the full `source_run_ids` list for both CSVs, the
per-backbone confusion-pair counts, and both family error-rate tables, so every
figure quoted above can be regenerated with
`.venv/bin/python scripts/person3/make_confusion_and_perclass.py`.
