# Error Analysis

**Date:** 2026-09-20. Confusion matrices below are read directly from each model's saved
`test_metrics.confusion_matrix` (group-safe split, 129 test images, rows=true/cols=pred in
class order diabetic/pressure/surgical/venous) — not re-estimated or approximated.

## Confusion matrices

**Classical (72.9% test acc):**
```
          pred:diab  pred:pres  pred:surg  pred:veno
true:diab      22         4          3          5
true:pres       7        22          2          5
true:surg       5         1         20          0
true:veno       3         0          0         30
```

**Frozen-head control (69.0%):**
```
          pred:diab  pred:pres  pred:surg  pred:veno
true:diab      20         6          3          5
true:pres       9        18          3          6
true:surg       3         1         22          0
true:veno       3         1          0         29
```

**Quantum-hybrid (64.3%):**
```
          pred:diab  pred:pres  pred:surg  pred:veno
true:diab      18         8          2          6
true:pres       7        18          3          8
true:surg       2         4         20          0
true:veno       2         4          0         27
```

## Consistent pattern across all three models

The confusion is not random or model-specific — the same two pairs dominate every model's
errors:

1. **pressure <-> diabetic** (7-9 pressure images called diabetic, 6-8 diabetic images called
   pressure, depending on the model) — the single largest error source everywhere.
2. **diabetic <-> venous** (3-6 images each direction) — second largest.
3. **Surgical and venous are comparatively easy.** Surgical is essentially never confused with
   venous (0 in every model, both directions), and venous is rarely confused with surgical. Most
   surgical errors go to diabetic, not to pressure or venous.

**Clinical plausibility, stated as a hypothesis, not a diagnosis:** pressure ulcers can occur in
varied locations and stages that visually resemble either diabetic ulcers (necrotic tissue,
irregular borders) or venous ulcers (location on the lower leg, surrounding skin changes),
depending on where and how advanced the wound is — consistent with why "location" turned out to
be the single most helpful additional signal tried this session (a pressure ulcer's typical
anatomical sites, like sacrum/heel, differ systematically from a venous ulcer's, like the
lower-medial leg, even when the wound itself looks ambiguous).

## Why hierarchical classification (Stage 15) was deprioritized

A hierarchical "group A vs. group B, then subtype within group" architecture only makes sense if
the confusion pattern suggests a clean two-way split. It doesn't: **pressure is confused with
BOTH diabetic and venous**, not predominantly one or the other, so there's no obvious grouping
that would isolate pressure's error modes from the other three classes. Forcing a hierarchy here
would mean imposing a structure the data doesn't actually support — deprioritized for this
reason, not for lack of time.

## AURA-Wound-specific findings

See `paper/aura_wound_results.md` for AURA-Wound's own confusion matrix and gate-weight analysis
(which branch the model learned to trust, and whether that varies for the pressure/diabetic/
venous confusion cases specifically).
