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
│   ├── extracted/                   ← Uncompressed versions of raw files, also contains trimmed versions
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
├── scripts/                           ← Added execution scripts
│   ├── train_icd9_predictor.py        ← For training the model
│   ├── evaluate_icd9_model.py         ← For model evaluation
│   └── predict_icd9_codes.py          ← For generating predictions
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

## 🔍 Source Data Details

This project uses the MIMIC-III (Medical Information Mart for Intensive Care III) clinical database, which contains de-identified health data associated with ~40,000 critical care patients. Three main source files are used:

### 1. NOTEEVENTS.csv
- Contains free-text clinical notes written by healthcare providers
- Includes discharge summaries, nursing notes, physician notes, etc.
- For this project, we focus specifically on **discharge summaries**
- Example structure:
  - ROW_ID: Unique identifier for each note
  - SUBJECT_ID: Unique identifier for each patient
  - HADM_ID: Unique identifier for each hospital admission
  - CHARTDATE: Date the note was charted
  - CATEGORY: Type of note (we filter for "Discharge summary")
  - TEXT: The actual clinical note text

### 2. DIAGNOSES_ICD.csv
- Contains ICD-9 diagnosis codes assigned to each hospital admission
- Example structure:
  - ROW_ID: Unique identifier for each diagnosis record
  - SUBJECT_ID: Unique identifier for each patient
  - HADM_ID: Unique identifier for each hospital admission
  - SEQ_NUM: Sequence number of the diagnosis
  - ICD9_CODE: The ICD-9 diagnosis code

### 3. D_ICD_DIAGNOSES.csv
- Provides descriptions for ICD-9 diagnosis codes
- Example structure:
  - ROW_ID: Unique identifier
  - ICD9_CODE: The ICD-9 diagnosis code
  - SHORT_TITLE: Brief description of the diagnosis
  - LONG_TITLE: Detailed description of the diagnosis

## 🧪 Preprocessing Pipeline

The preprocessing pipeline (`Pre-Processing.ipynb`) transforms raw MIMIC-III data into a format suitable for machine learning:

### 1. Data Loading and Initial Filtering
- Loads the three source files using DuckDB for efficient processing
- Filters NOTEEVENTS to include only discharge summaries
- Identifies the top 20 most frequent ICD-9 diagnosis codes

### 2. Text and Code Extraction
- Joins discharge summaries with their corresponding ICD-9 codes
- Creates a dataset linking each hospital admission to:
  - The full discharge summary text
  - All associated ICD-9 codes for that admission
  - The text descriptions of those codes

### 3. One-Hot Encoding of ICD-9 Codes
- Transforms the comma-separated list of ICD-9 codes into one-hot encoded columns
- Creates a binary feature for each of the top 20 ICD-9 codes
- Example: If admission has code "4019", the column "ICD9_4019" is set to 1, otherwise 0

### 4. Clinical Section Extraction
- Identifies and extracts key clinical sections from the discharge summaries
- Uses regular expressions to detect section headers like:
  - Chief Complaint
  - History of Present Illness
  - Discharge Diagnosis
  - Assessment
  - Hospital Course
- Creates separate columns for each extracted section
- Adds binary indicators for section presence/absence
- Computes section metrics (length, word count)

### 5. Text Preprocessing
- Standardizes de-identified elements (e.g., [**2151-7-16**] → [DATE])
- Removes excessive whitespace and normalizes formatting
- Preserves clinical content while reducing noise

### 6. Creation of Weighted Clinical Text
- Combines the most diagnostically relevant sections into a focused text representation
- Prioritizes sections most likely to contain coding-relevant information

## 📊 Dataset Structure

The final preprocessed dataset (`summary_results.csv`) contains:

### Patient Identifiers
- SUBJECT_ID: Unique patient identifier 
- HADM_ID: Unique hospital admission identifier

### Clinical Text Features
- summary_snippet_clean: The preprocessed complete discharge summary
- clinical_weighted_text: Combined text from key diagnostic sections
- section_X: Extracted text from specific clinical sections (e.g., section_CHIEF_COMPLAINT)

### Section Metadata
- has_X: Binary indicators for section presence (e.g., has_ASSESSMENT)
- section_X_length: Character count for each section
- section_X_words: Word count for each section

### Diagnosis Code Features
- ICD9_XXXX: One-hot encoded columns for each of the top 20 ICD-9 codes
- diagnosis_count: Total number of diagnosis codes for the admission

### Precision-Recall Performance
- Average precision scores by condition:
  - Code 4280 (Heart failure): AP = 0.83
  - Code 42731 (Atrial fibrillation): AP = 0.83
  - Code 5849 (Kidney failure): AP = 0.78
  - Code 41401 (Coronary atherosclerosis): AP = 0.53
  - Code 4019 (Hypertension): AP = 0.45

### Key Findings
- Model performs best on conditions with specific terminology in notes
- Heart failure and atrial fibrillation have clearest clinical language markers
- Common conditions like hypertension are more challenging to differentiate
- Section-specific features improved predictions for nuanced diagnoses
- Performance correlates with distinctiveness of clinical language for each condition

## 🧰 Running the Model

### 1. Training (`train_icd9_predictor.py`)

```bash
python train_icd9_predictor.py \
    --data_dir ../data \
    --output_dir ../models/bert_icd9 \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --text_column clinical_weighted_text
```

Final use case:

```bash
python train_icd9_predictor.py --data_dir ../data --output_dir ../models/bert_icd9 --batch_size 8 --epochs 3 --learning_rate 2e-5 --text_column clinical_weighted_text --use_section_texts --use_binary_indicators --use_numerical_features
```


Key parameters:
- `--data_dir`: Directory containing the data
- `--output_dir`: Directory to save the model
- `--batch_size`: Batch size for training (reduce if running into memory issues)
- `--epochs`: Number of training epochs
- `--text_column`: Which text column to use (clinical_weighted_text or summary_snippet_clean)

### 2. Evaluation (`evaluate_icd9_model.py`)

```bash
python evaluate_icd9_model.py \
    --model_dir ../models/bert_icd9 \
    --test_data_path ../data/preprocessed/summary_results.csv \
    --output_dir ../models/bert_icd9/evaluation_results \
    --threshold 0.5
```

```bash
python evaluate_icd9_model.py --model_dir ../models/bert_icd9 --test_data_path ../data/preprocessed/summary_results_trimmed.csv --output_dir ../models/bert_icd9/evaluation_results --text_column summary_snippet_clean
```

Key parameters:
- `--model_dir`: Directory containing the trained model
- `--test_data_path`: Path to the test data
- `--threshold`: Classification threshold (0.5 default)

### 3. Prediction (`predict_icd9_codes.py`)

```bash
# For a single file
python predict_icd9_codes.py \
    --model_dir ../models/bert_icd9 \
    --file ../docs/test_note.txt \
    --output_dir ../predictions

# Interactive mode
python predict_icd9_codes.py \
    --model_dir ../models/bert_icd9 \
    --interactive
```

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Top 20 ICD-9 Codes

The project focuses on predicting the 20 most common ICD-9 codes in the MIMIC-III dataset:

1. 4019: Unspecified essential hypertension
2. 4280: Congestive heart failure, unspecified
3. 42731: Atrial fibrillation
4. 41401: Coronary atherosclerosis of native coronary artery
5. 5849: Acute kidney failure, unspecified
6. 25000: Diabetes mellitus without mention of complication, type II or unspecified type
7. 2724: Other and unspecified hyperlipidemia
8. 51881: Acute respiratory failure
9. 5990: Urinary tract infection, site not specified
10. 53081: Esophageal reflux
11. 2720: Pure hypercholesterolemia
12. V053: Need for prophylactic vaccination and inoculation against viral hepatitis
13. V290: Observation for suspected infectious condition
14. 2859: Anemia, unspecified
15. 2449: Unspecified acquired hypothyroidism
16. 486: Pneumonia, organism unspecified
17. 2851: Acute posthemorrhagic anemia
18. 2762: Acidosis
19. 496: Chronic airway obstruction, not elsewhere classified
20. 99592: Severe sepsis
