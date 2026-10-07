# Multi-View Automated Grammar Scoring Engine for Spoken Audio

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/ASR-OpenAI_Whisper-00A67E?style=flat)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Deliverable**: Automated Grammar Scoring Engine predicting continuous Mean Opinion Score (MOS) in $[0.0, 5.0]$ from 45–60 second spoken audio clips.  
> **Key Benchmark**: **Out-Of-Fold CV RMSE: 0.6833** | **Pearson $r$: 0.8342** | **QWK: 0.7699**

---

## Executive Summary & Headline Results

This repository contains an end-to-end multi-view machine learning system for continuous grammar scoring of spoken English audio recordings. The solution integrates automated speech recognition (Whisper ASR), syntactic complexity analysis (spaCy), transformer acceptability scoring (CoLA RoBERTa), language perplexity (GPT-2), acoustic signal extraction (librosa), and dense semantic embeddings (SentenceTransformers).

Model predictions are dynamically combined using a non-negative Ridge Stacker over 4 distinct base estimators (Ridge, LightGBM, XGBoost, and SVR) trained under a 5-Fold Repeated Stratified Cross-Validation protocol.

### Benchmark Metrics

| Metric | Out-Of-Fold (OOF) CV Estimate | Full Training Set Refit |
| :--- | :---: | :---: |
| **RMSE (Primary Evaluation Metric)** | **0.6833** | **0.5172** |
| **MAE (Mean Absolute Error)** | **0.5331** | **0.3950** |
| **Pearson Correlation ($r$)** | **0.8342** | **0.9051** |
| **Spearman Rank Correlation ($\rho$)** | **0.7587** | **0.8410** |
| **$R^2$ Score** | **0.6955** | **0.8250** |
| **QWK (Quadratic Weighted Kappa)** | **0.7699** | **0.8710** |

---

## 3-View Architecture & Feature Pipeline

```
                              ┌─────────────────────────────────────────┐
                              │    Spoken Audio Clip (45 - 60 sec)      │
                              └────────────────────┬────────────────────┘
                                                   │
          ┌────────────────────────────────────────┼────────────────────────────────────────┐
          ▼                                        ▼                                        ▼
┌──────────────────┐                     ┌──────────────────┐                     ┌──────────────────┐
│  View A: Text &  │                     │ View B: Acoustic │                     │ View C: Semantic │
│    Syntactic     │                     │     & Timing     │                     │    Embeddings    │
├──────────────────┤                     ├──────────────────┤                     ├──────────────────┤
│• Whisper ASR     │                     │• WPM & Artic.    │                     │• SentenceTrans-  │
│• spaCy Dep Depth │                     │• Pause Rates     │                     │  formers         │
│• CoLA RoBERTa    │                     │• 13 MFCCs        │                     │  (all-MiniLM-L6) │
│• GPT-2 Perplex.  │                     │• RMS, ZCR        │                     │• Fold-level PCA  │
│• Filler Rates    │                     │• Centroid        │                     │  Compression     │
└────────┬─────────┘                     └────────┬─────────┘                     └────────┬─────────┘
         │                                        │                                        │
         └────────────────────────────────────────┼────────────────────────────────────────┘
                                                  │
                                                  ▼
                               ┌──────────────────────────────────┐
                               │  Cross-Validation & Ensembling   │
                               ├──────────────────────────────────┤
                               │ • 5-Fold Repeated Stratified CV  │
                               │ • Leakage-Free Scaler & Imputer  │
                               │ • Non-Negative Ridge Stacker     │
                               │ • Output Clipping to [0.0, 5.0]  │
                               └──────────────────┬───────────────┘
                                                  │
                                                  ▼
                               ┌──────────────────────────────────┐
                               │  outputs/submission.csv (216)    │
                               └──────────────────────────────────┘
```

1. **View A: Textual & Syntactic Complexity**
   - **ASR Transcription**: OpenAI Whisper (`tiny.en`) extracting raw transcripts and word-level timestamp alignments.
   - **Linguistic Error Heuristics**: Subject-verb agreement, double negation, and adjacent word repetition rates.
   - **Syntactic Parsing**: spaCy (`en_core_web_sm`) dependency tree depth, clause complexity, and POS tag distributions.
   - **Grammatical Acceptability**: Fine-tuned `roberta-base-CoLA` acceptability probabilities.
   - **Fluency Perplexity**: GPT-2 language model perplexity (`gpt2`).
   - **Disfluency Detection**: Filler word frequencies (`um`, `uh`, `er`, `ah`, `like`, `you know`).

2. **View B: Acoustic Signal & Speech Timing**
   - **Speech Tempo**: Words-per-minute (WPM), articulation rate (words per speech second).
   - **Silence & Pauses**: Pause counts (>0.3s and >0.5s gaps) and pause length statistics.
   - **Spectral Audio**: Librosa 13-band MFCC means/stds, Root-Mean-Square (RMS) energy, Zero Crossing Rate (ZCR), and Spectral Centroid.

3. **View C: Dense Text Embeddings**
   - Pretrained `SentenceTransformer('all-MiniLM-L6-v2')` generating 384-dimensional dense semantic vectors, compressed via in-fold PCA (16 components).

---

## Directory Structure

```
Grammer scoring engine/
│
├── grammar_scoring.ipynb      # Main submission notebook (End-to-End pipeline)
├── main.py                    # Modular CLI runner
├── requirements.txt           # Environment dependencies
├── README.md                  # Comprehensive technical documentation
│
├── src/                       # Modular Python Package
│   ├── __init__.py
│   ├── config.py              # Directory paths & global seeds
│   ├── eda.py                 # Data loader & target distribution audit
│   ├── asr.py                 # Whisper ASR transcription & caching
│   ├── features_text.py       # Linguistic, spaCy, CoLA & GPT-2 features
│   ├── features_audio.py      # Acoustic MFCCs & timing statistics
│   ├── embeddings.py          # SentenceTransformers text embeddings
│   ├── validation.py          # 5-Fold Repeated Stratified CV engine
│   ├── ensembling.py          # Non-negative Ridge stacker & ablation study
│   ├── evaluation.py          # Metric calculations & figures generator
│   ├── interpretability.py    # LightGBM feature importances & SHAP analysis
│   └── inference.py           # Submission generator & integrity assertions
│
├── outputs/
│   └── submission.csv         # Verified test set predictions (216 samples)
│
├── figures/                   # Generated visual plots
│   ├── label_distribution.png
│   ├── duration_hist.png
│   ├── true_vs_pred.png
│   ├── residuals.png
│   ├── ablation_chart.png
│   ├── feature_importance.png
│   └── shap_summary.png
│
└── cache/                     # Cached feature matrices & parquet files
    ├── transcripts.parquet
    ├── feats_text.parquet
    ├── feats_audio.parquet
    └── emb_text.npy
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.11+
- PyTorch 2.0+

### 2. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd "Grammer scoring engine"

# Create and activate virtual environment
python -m venv myenv
# Windows:
myenv\Scripts\activate
# Linux/macOS:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

---

## How to Run

### Option A: Running via Python CLI (Full Pipeline)

To run the complete pipeline end-to-end, execute:

```bash
python main.py
```

### Option B: Running via Jupyter Notebook

Launch Jupyter Notebook and open `grammar_scoring.ipynb`:

```bash
jupyter notebook grammar_scoring.ipynb
```

Execute all cells sequentially from top to bottom.

---

## Submission Verification & Integrity Assertions

The inference module (`src/inference.py`) automatically enforces strict validation checks on the output submission file (`outputs/submission.csv`):

- ✅ **Exact Sample Count**: Exactly 216 test predictions matching `test.csv`.
- ✅ **Null Check**: `0` NaN or missing values.
- ✅ **Bounded Range**: All predicted grammar scores bounded within $[0.0, 5.0]$.
- ✅ **Key Alignment**: Exact filename and split ordering preserved.

---

## License

This project is released under the **MIT License**.
