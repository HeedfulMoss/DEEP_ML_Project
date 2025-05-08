# ICD-9 Code Prediction from Clinical Text

This project provides a deep learning solution for automatically predicting ICD-9 diagnosis codes from clinical discharge summaries. It implements a BERT-CNN architecture to analyze clinical text and predict which ICD-9 codes are present.

## Architecture Overview

The model uses a combination of BERT (Bidirectional Encoder Representations from Transformers) and Convolutional Neural Networks (CNNs) for multi-label classification of ICD-9 codes:

1. **BERT Encoder**: Extracts rich contextual representations from clinical text
2. **Multi-scale CNN Layers**: Capture n-gram patterns of different sizes from BERT embeddings
3. **Global Max Pooling**: Extract the most salient features across the sequence
4. **Fully Connected Layer**: Final classification for each ICD-9 code

## Directory Structure

The project follows the structure outlined in the original README:

```
DEEP_ML_Project/
├── data/
│   ├── raw/
│   ├── extracted/
│   └── preprocessed/
│       ├── summary_results.csv
│       └── summary_results_trimmed.csv
│
├── models/
│   └── bert_icd9/
│       ├── icd9_bert_model/
│       ├── results/
│       └── icd9_label_binarizer.pkl
│
├── notebooks/
│
└── scripts/
    ├── train_icd9_predictor.py
    ├── evaluate_icd9_model.py
    └── predict_icd9_codes.py
```

## Scripts

### 1. Training (`train_icd9_predictor.py`)

This script trains the BERT-CNN model on the MIMIC dataset:

```bash
python train_icd9_predictor.py \
    --data_dir ../data \
    --output_dir ../models/bert_icd9 \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --text_column clinical_weighted_text
```

or 

```bash
python train_icd9_predictor.py --data_dir ../data --output_dir ../models/bert_icd9 --batch_size 8 --epochs 3 --learning_rate 2e-5 --text_column clinical_weighted_text --use_section_texts --use_binary_in
```

Key parameters:
- `--data_dir`: Directory containing the data
- `--output_dir`: Directory to save the model
- `--batch_size`: Batch size for training (reduce if running into memory issues)
- `--epochs`: Number of training epochs
- `--text_column`: Which text column to use (clinical_weighted_text or summary_snippet_clean)

**Quick Start:** To ensure the code runs without errors, start with a small number of epochs.

### 2. Evaluation (`evaluate_icd9_model.py`)

This script evaluates a trained model on test data:

```bash
python evaluate_icd9_model.py \
    --model_dir ../models/bert_icd9 \
    --test_data_path ../data/preprocessed/summary_results.csv \
    --output_dir ../models/bert_icd9/evaluation_results \
    --threshold 0.5
```

or 
```bash
python evaluate_icd9_model.py --model_dir ../models/bert_icd9 --test_data_path ../data/preprocessed summary_results_trimmed.csv --output_dir ../models/bert_icd9/evaluation_results --text_column summary_snippet_clean
```


Key parameters:
- `--model_dir`: Directory containing the trained model
- `--test_data_path`: Path to the test data
- `--threshold`: Classification threshold (0.5 default)

### 3. Prediction (`predict_icd9_codes.py`)

This script allows predicting ICD-9 codes for new texts:

```bash
# For a single file
python predict_icd9_codes.py \
    --model_dir ../models/bert_icd9 \
    --file ../docs/test_note.txt \
    --output_dir ../predictions

# For a CSV file
python predict_icd9_codes.py \
    --model_dir ../models/bert_icd9 \
    --csv ../data/new_notes.csv \
    --text_column note_text \
    --output_dir ../predictions

# Interactive mode
python predict_icd9_codes.py \
    --model_dir ../models/bert_icd9 \
    --interactive
```

## Implementation Details

### Model Architecture Rationale

1. **BERT Base Layer**
   - Pretrained BERT provides deep contextual understanding of medical text
   - Captures semantic relationships and domain-specific terminology
   - Base uncased model balances performance with computational efficiency

2. **CNN Layers (Why CNN?)**
   - Multiple kernel sizes (3, 4, 5) capture different n-gram patterns
   - Efficiently detects key phrases and medical terminology
   - Lighter and faster than using recurrent networks on BERT outputs
   - Clinical coding often relies on specific keyword patterns that CNNs excel at identifying

3. **Global Max Pooling**
   - Extracts the most important features across the sequence
   - Reduces dimensionality while preserving key information
   - Makes the model robust to varying text lengths

4. **Multi-label Binary Classification**
   - Independent sigmoid outputs for each ICD-9 code
   - Allows each code to be predicted independently
   - Better suited for the multi-label nature of medical coding than softmax

### Data Handling

The model works with the pre-processed MIMIC dataset:
- Uses DuckDB for efficient handling of large CSV files
- Verifies dataset format against the provided template
- Implements dynamic batch loading to handle memory constraints

### Training Process

- Uses AdamW optimizer with weight decay for regularization
- Implements linear warm-up and learning rate scheduling
- Early stopping based on validation AUC to prevent overfitting
- Automatic checkpointing and results logging

### Evaluation Metrics

The model provides comprehensive evaluation metrics:
- AUC-ROC scores (overall and per-code)
- Precision, recall, and F1 score (micro and macro)
- Confusion matrices for top codes
- Threshold analysis for optimal decision boundaries

## Performance Optimization

The code includes several optimizations for GPU training:
- Dynamic batch sizing based on available memory
- Gradient accumulation for effective larger batch sizes
- Mixed precision training support
- Efficient data loading with proper worker configuration

## Troubleshooting

**Memory Issues**: If running into CUDA out of memory errors:
1. Reduce batch size (--batch_size 4 or lower)
2. Reduce sequence length (--max_length 256)
3. Use gradient accumulation (automatically enabled with small batch sizes)

**File Loading Issues**: 
- Make sure to use the exact paths as specified in the README
- The code includes verification against the trimmed dataset template

## Requirements

- PyTorch >= 1.10.0
- Transformers >= 4.15.0
- DuckDB >= 0.3.3
- scikit-learn >= 1.0.0
- pandas, numpy, matplotlib, seaborn