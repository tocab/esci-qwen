# Teaching Qwen to understand product relevance

**Qwen3.5-0.8B-Base · Amazon ESCI · LoRA**

## Model flow

Shopping query + product details → frozen Qwen backbone + **10.82M trainable
LoRA parameters** → probabilities for **Exact / Substitute / Complement / Irrelevant**.
All four answer candidates are scored; training uses cross-entropy.

## Data

| Training | Validation | Official test |
|---:|---:|---:|
| 202,359 | 19,458 | 638,016 |

ESCI Task 2, large subset. US English, Spanish, and Japanese.
Query-ID / normalized-text grouping; overlapping training groups quarantined.

## Fine-tuning

**One RTX 3090, 24 GB VRAM · 3 epochs · 34.5 training hours**

LoRA rank **16** · effective batch **32** · token limit **256** · peak learning
rate **1e-4**. bf16, AdamW, hourly saves, and approximately six-hour validation.
Stopped automatically when validation improvement plateaued.

## Validation checkpoint selection

**Best macro-F1: 0.5570 · Accuracy: 70.04% · Best checkpoint: epoch 2**

| Exact F1 | Substitute F1 | Complement F1 | Irrelevant F1 |
|---:|---:|---:|---:|
| 0.819 | 0.564 | 0.388 | 0.457 |

## Full official test comparison

All **638,016** official test examples were evaluated.

| Model | Macro F1 | Accuracy |
|---|---:|---:|
| Qwen3.5-0.8B + LoRA | **54.69%** | **70.05%** |
| CLEF Flash 9B | **57.01%** | **67.86%** |

CLEF leads macro F1 by **2.32 percentage points**. Qwen has higher accuracy,
but Exact matches dominate the test set; macro F1 is the primary balanced metric.
Both received identical retained product information through native interfaces.
This does not reproduce the published Decision Index benchmark.

Training used a six-epoch learning-rate schedule and stopped after three epochs;
validation timing and results can vary across hardware.

[Illustrated PDF](assets/overview.pdf) · [Detailed results](results/comparison.json)

*Experiment overview · 14 October 2026. Official test evaluation complete.*
