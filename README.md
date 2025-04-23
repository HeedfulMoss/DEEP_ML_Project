# DEEP_ML_Project

## Project Tree Summary
```
DEEP_ML_Project/
├── .git/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├──data/
│   ├── raw/                         ← Original compressed files from MIMIC-III
│   │   ├── D_ICD_DIAGNOSES.csv.gz
│   │   ├── DIAGNOSES_ICD.csv.gz
│   │   ├── NOTEEVENTS.csv.gz       🔒 [Manual Download Required]
│   │
│   ├── extracted/                   ← Uncompressed versions of raw files
│   │   ├── D_ICD_DIAGNOSES.csv
│   │   ├── DIAGNOSES_ICD.csv
│   │   ├── NOTEEVENTS.csv          🔒 [Manual Download Required]
│   │
│   └── preprocessed/                   ← Pre-cleaned, ML-ready data
│       ├── summary_results.csv            ❌ ← Created by `Pre-Processing.ipynb`
│       ├── summary_results_trimmed.csv  ← Created by `Pre-Processing.ipynb`
│
├── models/
│   ├── bert_icd9/                  ← directory to store files generated from MIMIC_ICD9_Bert_Train
│       ├── icd9_bert_model/        ❌ ← [HuggingFace model artifacts] too large
│       ├── results/                ❌   
│       └── icd9_label_binarizer.pkl
│   
├── notebooks/
│   ├── Pre-Processing.ipynb           ← pre-processing, creates the summary_results files
│   ├── MIMIC_ICD9_Bert_Train.ipynb     ← model training, creates populates "bert_icd9" 
│
├── docs/
│   ├── next_steps.txt
│   └── test_note.txt
│
└── venv/                       ❌ [Ignored virtual environment]
```
**Legend**  
❌ Ignored (see `.gitignore`)  
🔒 Requires manual download (not included in repo)  

## Important setup notes:

| Path | Reason |
|------|--------|
| `venv/` | Local Python virtual environment |
| `data/raw/NOTEEVENTS.csv.gz` | Large original MIMIC data Notes file |
| `data/extracted/NOTEEVENTS.csv` | Uncompressed MIMIC data Notes CSV from raw data |
| `data/preprocessed/summary_results.csv` | Generated from `Pre-Processing.ipynb` |
| `models/bert_icd9/icd9_bert_model/` | HuggingFace model artifacts |
| `models/bert_icd9/results/` | Training logs, metrics, evaluation results |
| `.ipynb_checkpoints/` | Jupyter notebook auto-saves |

## 📖 Project Overview

**DEEP_ML_Project** is an end-to-end deep learning pipeline built to train and evaluate transformer models on the MIMIC-III clinical dataset. It includes:

- 🧼 Preprocessing of clinical notes and diagnosis codes  
- 🧠 Fine-tuning BERT for ICD9 code classification  
- 📊 Generating model evaluation reports and predictions  
- 💾 Organized modular output structure for easy tracking  

## 🧰 Core Notebooks

| Notebook | Description |
|----------|-------------|
| `Pre-Processing.ipynb` | Prepares and merges MIMIC data into a training-ready format |
| `MIMIC_ICD9_Bert_Train.ipynb` | Fine-tunes BERT model and outputs model artifacts + logs |

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt