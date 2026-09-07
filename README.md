# Biomedical RAG Hallucination Reduction Dissertation

## Overview

This repository contains the experimental pipeline for a Master's dissertation investigating whether Retrieval-Augmented Generation (RAG) reduces hallucinations in biomedical question answering, and whether a more complex RAG pipeline provides additional benefits over a simpler dense-retrieval system.

The project compares four controlled generation conditions on the same biomedical QA benchmark:

1. **Baseline LLM** — standalone generator with no retrieved evidence
2. **Basic RAG** — dense retrieval using `sentence-transformers/all-MiniLM-L6-v2`
3. **Advanced RAG** — BM25 + dense retrieval + weighted reciprocal rank fusion + cross-encoder reranking
4. **Gold-context control** — the same generator supplied directly with the benchmark-attached evidence

The final experiment uses a frozen 500-question test set and paired statistical analysis across systems.

\---

## Research Question

> To what extent does Retrieval-Augmented Generation reduce hallucinations in biomedical/healthcare question answering compared with a standalone LLM, and does enhanced RAG provide additional improvements over basic RAG?

\---

## Project Goals

The project evaluates three related questions:

* Does RAG improve biomedical answer decision performance?
* Does RAG reduce unsupported or contradicted biomedical factual claims?
* Does an enhanced hybrid + reranking retrieval pipeline outperform a simpler dense-retrieval RAG system?

The experimental design deliberately separates system development, protocol freezing, final test generation, final evaluation, and exploratory post-hoc analysis.

\---

## Dataset

The project uses a PubMedQA-style biomedical QA dataset containing **1,000 records**.

The data was split into:

* **500 development questions**
* **500 frozen final-test questions**

The retrieval corpus contains **3,358 passages**.

### Leakage Control

The generator and retrieval systems were not given evaluation targets such as:

* `LONG\_ANSWER`
* `final\_decision`
* `reasoning\_required\_pred`
* `reasoning\_free\_pred`
* reference answers
* gold decisions

Gold context IDs were used only for evaluation and for the intentionally oracle-style Gold-context control.

\---

## System Architectures

### Baseline LLM

Receives the biomedical question and frozen prompt, but no external evidence.

### Basic RAG

Uses dense retrieval with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Pipeline:

```text
Question
  ↓
Dense embedding
  ↓
Cosine-similarity retrieval
  ↓
Top-10 candidates
  ↓
Top-5 contexts
  ↓
Generator
```

### Advanced RAG

Uses:

* BM25 retrieval
* dense retrieval
* weighted Reciprocal Rank Fusion
* candidate union
* cross-encoder reranking
* top-5 contexts for generation

Frozen retrieval settings:

```text
BM25 weight = 0.75
Dense weight = 1.0
RRF k = 60
```

Cross-encoder:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

### Gold-Context Control

Uses the same generator as the other systems but is supplied with the benchmark-attached evidence passages directly.

\---

## Generator

The frozen generator used for the final experiment was:

```text
gpt-5.4-mini-2026-03-17
```

The model returns structured JSON with:

```json
{
  "decision": "yes | no | maybe",
  "answer": "concise biomedical answer",
  "citations": \["C1", "C2"]
}
```

For the baseline condition, citations are empty.

\---

## Hallucination Evaluation

The frozen blinded evaluator used:

```text
Judge version: v3\_final
Judge model: gpt-5.4-2026-03-05
Reasoning effort: low
Seed: 42
```

The evaluator:

1. extracts externally verifiable biomedical factual claims,
2. excludes abstention, uncertainty, evidence-availability statements, citation-format commentary, and non-biomedical meta statements,
3. labels each scored claim as supported, unsupported, or contradicted,
4. aggregates claim-level and answer-level metrics.

### Primary Hallucination Metric

```text
Hallucination rate
=
(unsupported claims + contradicted claims)
/
total scored biomedical factual claims
```

### Groundedness

```text
Groundedness
=
supported claims
/
total scored biomedical factual claims
```

### Contradiction Rate

```text
Contradiction rate
=
contradicted claims
/
total scored biomedical factual claims
```

\---

## Human Validation

The automated hallucination evaluator was assessed using:

* claim-label validation
* system-balanced binary validation
* claim extraction / exclusion validation

The human-calibrated hallucination estimate is treated as a **sensitivity analysis only** and does not replace the frozen V3 primary results.

\---

## Retrieval Evaluation

Retrieval was evaluated using:

* **Source Hit@k**
* **MRR@10**
* **Gold-context Recall@k**

\---

## Statistical Analysis

The same 500 final-test questions were used for all four systems, enabling paired comparisons.

### Decision Evaluation

* accuracy
* macro-F1
* paired bootstrap confidence intervals
* exact McNemar tests
* Holm correction

### Hallucination Evaluation

* micro claim-level hallucination rate
* question-cluster bootstrap confidence intervals
* paired answer-level Wilcoxon tests
* Holm correction

### Retrieval Evaluation

* exact McNemar tests for Source Hit@k
* paired bootstrap confidence intervals
* paired Wilcoxon tests
* Holm correction

\---

## Final Test Results

### Decision Performance

|System|Accuracy|Macro-F1|
|-|-:|-:|
|Baseline LLM|32.20%|0.2888|
|Basic RAG|62.20%|0.5385|
|Advanced RAG|60.60%|0.5147|
|Gold-context control|72.20%|0.5791|

### V3 Hallucination Results

|System|Hallucination Rate|Groundedness|Contradiction Rate|
|-|-:|-:|-:|
|Baseline LLM|61.08%|38.92%|5.82%|
|Basic RAG|8.49%|91.51%|1.82%|
|Advanced RAG|7.85%|92.15%|1.70%|
|Gold-context control|4.30%|95.70%|0.13%|

### Human-Calibrated Sensitivity Analysis

|System|Calibrated Hallucination|95% CI|
|-|-:|-:|
|Baseline LLM|32.49%|18.32%–46.65%|
|Basic RAG|2.97%|1.27%–4.67%|
|Advanced RAG|3.53%|1.96%–5.10%|
|Gold-context control|1.51%|0.65%–2.37%|

### Retrieval Performance

|Metric|Basic RAG|Advanced RAG|
|-|-:|-:|
|Source Hit@1|96.80%|96.60%|
|Source Hit@3|98.40%|99.00%|
|Source Hit@5|99.00%|99.00%|
|Source Hit@10|99.40%|99.20%|
|MRR@10|0.9769|0.9769|
|Gold Recall@5|77.47%|77.54%|

\---

## Main Findings

* RAG substantially improved biomedical decision performance over the standalone LLM.
* RAG sharply reduced hallucination.
* Basic RAG reduced the V3 hallucination rate from **61.08% to 8.49%**.
* Advanced RAG reduced it to **7.85%**.
* Advanced RAG did **not** show a statistically detectable advantage over Basic RAG on the final test set.
* Both RAG systems reached **99.0% Source Hit@5**, indicating near-ceiling correct-source retrieval at the generation cutoff.
* The Gold-context control still made errors, showing that residual error cannot be explained by retrieval alone.

\---

## Exploratory Error Analysis

### Basic RAG

```text
Decision errors:                         189
Decision errors despite Source Hit@5:   187
Retrieval misses @5:                      5
Answers with hallucinated claims:        93
Hallucination despite Source Hit@5:      90
```

### Advanced RAG

```text
Decision errors:                         197
Decision errors despite Source Hit@5:   193
Retrieval misses @5:                      5
Answers with hallucinated claims:        87
Hallucination despite Source Hit@5:      85
```

This analysis is exploratory and should not be interpreted as a causal decomposition of error mechanisms.

\---

## Repository Structure

```text
biomedical\_rag\_dissertation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── 01\_dataset\_inspection.py
│   ├── 02\_dataset\_analysis.py
│   ├── 03\_record\_inspection.py
│   ├── 04\_prepare\_data.py
│   ├── 05\_bm25\_retrieval.py
│   ├── 06\_dense\_retrieval.py
│   ├── ...
│   ├── 32\_freeze\_final\_protocol.py
│   ├── ...
│   ├── 44\_final\_test\_retrieval\_evaluation.py
│   ├── 45\_final\_test\_retrieval\_statistics.py
│   ├── 46\_integrated\_final\_results.py
│   ├── 47\_exploratory\_error\_analysis.py
│   ├── 48\_prepare\_dissertation\_tables.py
│   └── 49\_prepare\_dissertation\_figures.py
│
├── results/
│   ├── frozen\_protocol/
│   ├── test/
│   │   └── evaluation/
│   ├── dissertation\_tables/
│   └── dissertation\_figures/
│
├── requirements.txt
└── README.md
```

\---

## Installation

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

\---

## API Configuration

Set the OpenAI API key before running API-dependent scripts.

Windows PowerShell:

```powershell
$env:OPENAI\_API\_KEY="your-key-here"
```

macOS / Linux:

```bash
export OPENAI\_API\_KEY="your-key-here"
```

Do not commit API keys to Git.

\---

## Important Final Scripts

### Integrated final scorecard

```powershell
python src/46\_integrated\_final\_results.py
```

### Exploratory error analysis

```powershell
python src/47\_exploratory\_error\_analysis.py
```

### Dissertation tables

```powershell
python src/48\_prepare\_dissertation\_tables.py
```

### Dissertation figures

```powershell
python src/49\_prepare\_dissertation\_figures.py
```

\---

## Dissertation Figures and Flowcharts

The dissertation uses methodology diagrams covering:

* main dissertation workflow
* methodological refinement workflow
* dataset preparation and leakage control
* Basic vs Advanced RAG retrieval architecture
* controlled four-system generation experiment
* hallucination evaluation and human validation
* final test evaluation and reporting

The Results chapter includes plots for:

* decision performance
* V3 hallucination rate
* human-calibrated hallucination sensitivity
* Source Hit@k
* Gold-context Recall@k
* exploratory residual errors

\---

## Reproducibility Safeguards

The experiment uses several safeguards:

* same 500 final-test questions for all systems
* system settings frozen before final-test evaluation
* no post-test retrieval tuning
* no prompt changes after protocol freeze
* no generator changes after protocol freeze
* no V3 evaluator changes after protocol freeze
* primary metrics fixed before final evaluation
* Gold-context control not assumed to be perfect
* Advanced RAG not assumed to be superior
* exploratory analyses clearly separated from confirmatory analyses

\---

## Limitations

Important limitations include:

* benchmark literature spans older biomedical publications rather than current 2026 medicine,
* automated hallucination evaluation is imperfect,
* human validation was performed by a single annotator,
* Source Hit@k is a coarse retrieval-success metric,
* successful source retrieval does not guarantee that all answer-bearing evidence was retrieved,
* the experiment evaluates benchmark biomedical QA rather than real clinical decision-making,
* findings depend on the frozen models, prompts, retrieval settings, and evaluator used in this study.

\---

## Conclusion

The experiment supports the conclusion that Retrieval-Augmented Generation can substantially reduce unsupported biomedical factual generation while also improving answer decision performance.

However, additional retrieval complexity did not produce a statistically detectable improvement over the simpler dense-retrieval Basic RAG pipeline on the final test set.

The Gold-context control further showed that residual errors remain even when benchmark evidence is supplied directly, suggesting that biomedical QA error is not solely a retrieval problem.

\---

## Academic Use

This repository was developed as part of a Master's dissertation project.

When adapting this work, clearly distinguish between:

* confirmatory analyses,
* sensitivity analyses,
* exploratory analyses.

\---

## License

\## License



This project's source code is released under the MIT License.



See the \[LICENSE](LICENSE) file for details.

\---

## Author

Aniruddha Patil

