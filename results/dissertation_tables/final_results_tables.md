# Final Test Results Tables

## Table 1. Primary system performance

| System | Accuracy | Macro-F1 | V3 hallucination | Groundedness | Contradiction | Human-calibrated hallucination |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LLM | 32.20% | 0.2888 | 61.08% | 38.92% | 5.82% | 32.49% |
| Basic RAG | 62.20% | 0.5385 | 8.49% | 91.51% | 1.82% | 2.97% |
| Advanced RAG | 60.60% | 0.5147 | 7.85% | 92.15% | 1.70% | 3.53% |
| Gold-context control | 72.20% | 0.5791 | 4.30% | 95.70% | 0.13% | 1.51% |

## Table 2. Retrieval performance

| System | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Gold Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| Basic RAG | 96.80% | 98.40% | 99.00% | 99.40% | 0.9769 | 77.47% |
| Advanced RAG | 96.60% | 99.00% | 99.00% | 99.20% | 0.9769 | 77.54% |

## Key confirmatory findings

- Basic RAG increased decision accuracy from 32.20% to 62.20% relative to the standalone LLM and reduced the frozen V3 micro-claim hallucination rate from 61.08% to 8.49%.
- Advanced RAG achieved 60.60% decision accuracy and a 7.85% V3 hallucination rate.
- Advanced RAG did not show a statistically detectable decision-accuracy advantage over Basic RAG: difference -1.60 percentage points, 95% CI -5.40 to +2.40 percentage points.
- Advanced RAG did not show a statistically detectable hallucination-rate advantage over Basic RAG: difference -0.64 percentage points, 95% cluster-bootstrap CI -2.33 to +1.08 percentage points.
- At the generation cutoff, Source Hit@5 was 99.00% for both Basic and Advanced RAG.

## Exploratory residual-error summary

- **Basic RAG:** 189 decision errors; 187 occurred despite Source Hit@5. Mean Gold-context Recall@5 was 0.8097 for correct decisions and 0.7171 for incorrect decisions.
- **Advanced RAG:** 197 decision errors; 193 occurred despite Source Hit@5. Mean Gold-context Recall@5 was 0.8275 for correct decisions and 0.6951 for incorrect decisions.

*The residual-error analysis is exploratory and should not be presented as a pre-specified confirmatory endpoint.*
