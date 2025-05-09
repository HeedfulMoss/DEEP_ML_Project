import os
import sys
import json
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import duckdb
import argparse
import logging
from tqdm import tqdm

# Import model class from training script
#from train_icd9_predictor import MIMIC_ICD9_Dataset, BERT_CNN_ICD9, predict_icd9_codes
from train_icd9_predictor import MIMIC_ICD9_Enhanced_Dataset as MIMIC_ICD9_Dataset, Enhanced_BERT_CNN_ICD9 as BERT_CNN_ICD9

# Try to import ultra-enhanced model classes
try:
    from ultra_enhanced_train_icd9_predictor import MIMIC_ICD9_Ultra_Enhanced_Dataset, Ultra_Enhanced_BERT_CNN_ICD9
    HAS_ULTRA_ENHANCED = True
    logger.info("Ultra-enhanced model classes loaded successfully")
except ImportError:
    HAS_ULTRA_ENHANCED = False
    logger.warning("Ultra-enhanced model classes not found. Will use regular enhanced model.")

def predict_icd9_codes(text, model, tokenizer, icd9_codes, device, max_length=512, threshold=0.5):
    """
    Predict ICD-9 codes for a given text.
    
    Args:
        text: Clinical text
        model: The trained model
        tokenizer: The BERT tokenizer
        icd9_codes: List of ICD-9 codes
        device: Device to run inference on
        max_length: Maximum sequence length
        threshold: Threshold for binary classification
    
    Returns:
        dict: Predicted ICD-9 codes and probabilities
    """
    # Encode the text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Check if this is an ultra-enhanced model
    is_ultra_enhanced = hasattr(model, 'medical_concept_layer') and model.medical_concept_layer is not None
    has_custom_thresholds = hasattr(model, 'get_thresholds')
    
    # Get custom thresholds if available
    custom_thresholds = None
    if has_custom_thresholds:
        custom_thresholds = model.get_thresholds().cpu().numpy()

    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Get predicted codes (above threshold)
    predicted_indices = np.where(probs >= threshold)[0]
    predicted_probs = probs[predicted_indices]
    
    # Create result dict
    result = {
        'predicted_codes': [],
        'all_probabilities': {}
    }
    
    # Add predicted codes
    for idx, prob in zip(predicted_indices, predicted_probs):
        code = icd9_codes[idx]
        result['predicted_codes'].append({
            'code': code,
            'probability': float(prob)
        })
    
    # Add all probabilities
    for i, code in enumerate(icd9_codes):
        result['all_probabilities'][code] = float(probs[i])
    
    # Sort predicted codes by probability (descending)
    result['predicted_codes'] = sorted(
        result['predicted_codes'],
        key=lambda x: x['probability'],
        reverse=True
    )
    
    return result

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("icd9_model_evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_dir, num_labels, device):
    """
    Load the trained model and tokenizer.
    
    Args:
        model_dir: Directory containing the model files
        num_labels: Number of ICD-9 codes to predict
        device: Device to load the model on
    
    Returns:
        tuple: Loaded model and tokenizer
    """
    # Find the model path
    model_path = None
    for file in os.listdir(model_dir):
        if file.endswith(".pt"):
            model_path = Path(model_dir) / file
            break
    
    if model_path is None:
        # Check for subfolders
        for subfolder in os.listdir(model_dir):
            subfolder_path = Path(model_dir) / subfolder
            if subfolder_path.is_dir():
                for file in os.listdir(subfolder_path):
                    if file == "best_model.pt":
                        model_path = subfolder_path / file
                        break
                if model_path is not None:
                    break
    
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    
    # Load tokenizer
    tokenizer_path = Path(model_dir) / "tokenizer"
    if not tokenizer_path.exists():
        # Look in subfolders
        for subfolder in os.listdir(model_dir):
            subfolder_path = Path(model_dir) / subfolder
            if subfolder_path.is_dir() and (subfolder_path / "tokenizer").exists():
                tokenizer_path = subfolder_path / "tokenizer"
                break
    
    if tokenizer_path.exists():
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.warning("Tokenizer not found, using default BERT tokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
    # Try to find model configuration information
    model_info_path = None
    run_dirs = []
    
    # Check for model_info.json in the model directory
    if Path(model_dir).joinpath("model_info.json").exists():
        model_info_path = Path(model_dir).joinpath("model_info.json")
    else:
        # Check in subdirectories (run directories)
        for item in os.listdir(model_dir):
            if Path(model_dir).joinpath(item).is_dir():
                if "run" in item or "enhanced" in item:
                    run_dirs.append(Path(model_dir).joinpath(item))
        
        # Look for model_info.json in run directories
        for run_dir in run_dirs:
            if run_dir.joinpath("model_info.json").exists():
                model_info_path = run_dir.joinpath("model_info.json")
                break
    
    # Default values
    section_names = []
    num_binary_features = 0
    num_numerical_features = 0
    
    # Load model configuration if found
    if model_info_path:
        logger.info(f"Loading model configuration from {model_info_path}")
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
        
        section_names = model_info.get("section_names", [])
        num_binary_features = len(model_info.get("binary_columns", []))
        num_numerical_features = len(model_info.get("numerical_columns", []))
        
        logger.info(f"Model configuration: {len(section_names)} sections, {num_binary_features} binary features, {num_numerical_features} numerical features")
    else:
        logger.warning("No model configuration found, using default values")
    
    # Initialize model with the configuration
    model = BERT_CNN_ICD9(
        num_labels=num_labels,
        section_names=section_names,
        num_binary_features=num_binary_features,
        num_numerical_features=num_numerical_features
    )
    
    # Load model weights
    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer

def load_icd9_codes(model_dir):
    """
    Load the ICD-9 codes and label binarizer.
    
    Args:
        model_dir: Directory containing the model files
    
    Returns:
        list: List of ICD-9 codes
    """
    # Try to load from label binarizer
    label_binarizer_path = Path(model_dir) / "icd9_label_binarizer.pkl"
    
    if not label_binarizer_path.exists():
        # Check parent directory
        label_binarizer_path = Path(model_dir).parent / "icd9_label_binarizer.pkl"
    
    if label_binarizer_path.exists():
        with open(label_binarizer_path, 'rb') as f:
            label_binarizer = pickle.load(f)
        icd9_codes = [label_binarizer['index_to_code'][i] for i in range(len(label_binarizer['index_to_code']))]
        return icd9_codes
    
    # Try to load from ICD-9 codes file
    icd9_codes_path = Path(model_dir) / "icd9_codes.json"
    if icd9_codes_path.exists():
        with open(icd9_codes_path, 'r') as f:
            icd9_codes = json.load(f)
        return icd9_codes
    
    # Try to find in subfolders
    for subfolder in os.listdir(model_dir):
        subfolder_path = Path(model_dir) / subfolder
        if subfolder_path.is_dir():
            for file in os.listdir(subfolder_path):
                if file == "icd9_codes.json":
                    with open(subfolder_path / file, 'r') as f:
                        icd9_codes = json.load(f)
                    return icd9_codes
    
    raise FileNotFoundError(f"No ICD-9 codes file found in {model_dir}")

def plot_roc_curves(y_true, y_pred, icd9_codes, top_n=5, save_path=None):
    """
    Plot ROC curves for the top N most frequent ICD-9 codes.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        icd9_codes: List of ICD-9 codes
        top_n: Number of top codes to plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate frequency of each code
    code_frequency = np.mean(y_true, axis=0)
    
    # Get indices of top N most frequent codes
    top_indices = np.argsort(code_frequency)[-top_n:][::-1]
    
    # Plot ROC curve for each top code
    for i in top_indices:
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{icd9_codes[i]} (AUC = {roc_auc:.2f})')
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot attributes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Top {top_n} Most Frequent ICD-9 Codes')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path / "roc_curves.png")
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curves(y_true, y_pred, icd9_codes, top_n=5, save_path=None):
    """
    Plot precision-recall curves for the top N most frequent ICD-9 codes.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        icd9_codes: List of ICD-9 codes
        top_n: Number of top codes to plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate frequency of each code
    code_frequency = np.mean(y_true, axis=0)
    
    # Get indices of top N most frequent codes
    top_indices = np.argsort(code_frequency)[-top_n:][::-1]
    
    # Plot precision-recall curve for each top code
    for i in top_indices:
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_pred[:, i])
        plt.plot(recall, precision, lw=2, label=f'{icd9_codes[i]} (AP = {avg_precision:.2f})')
    
    # Set plot attributes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves for Top {top_n} Most Frequent ICD-9 Codes')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path / "precision_recall_curves.png")
        plt.close()
    else:
        plt.show()

def plot_threshold_analysis(y_true, y_pred, save_path=None):
    """
    Plot the effect of different threshold values on precision, recall, and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Define threshold range
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    # Initialize metrics arrays
    precisions = []
    recalls = []
    f1_scores = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate macro-averaged metrics
        precision = np.mean([
            np.sum((y_pred_binary[:, i] == 1) & (y_true[:, i] == 1)) / max(np.sum(y_pred_binary[:, i] == 1), 1)
            for i in range(y_true.shape[1])
        ])
        
        recall = np.mean([
            np.sum((y_pred_binary[:, i] == 1) & (y_true[:, i] == 1)) / max(np.sum(y_true[:, i] == 1), 1)
            for i in range(y_true.shape[1])
        ])
        
        # Calculate F1-score
        f1 = 2 * (precision * recall) / max((precision + recall), 1e-8)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Plot metrics
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'r-', label='Recall')
    plt.plot(thresholds, f1_scores, 'g-', label='F1-score')
    
    # Set plot attributes
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Effect of Threshold on Classification Metrics')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path / "threshold_analysis.png")
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, icd9_codes, top_n=10, threshold=0.5, save_path=None):
    """
    Plot confusion matrices for the top N most frequent ICD-9 codes.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        icd9_codes: List of ICD-9 codes
        top_n: Number of top codes to plot
        threshold: Threshold for binary classification
        save_path: Path to save the plots
    """
    # Convert to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate frequency of each code
    code_frequency = np.mean(y_true, axis=0)
    
    # Get indices of top N most frequent codes
    top_indices = np.argsort(code_frequency)[-top_n:][::-1]
    
    # Create directory for confusion matrices if save_path is specified
    if save_path:
        cm_dir = save_path / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
    
    # Plot confusion matrix for each top code
    for i in top_indices:
        plt.figure(figsize=(8, 6))
        
        # Calculate confusion matrix
        tn = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0))
        tp = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1))
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        
        # Calculate metrics
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
        
        # Set plot attributes
        plt.title(f'Confusion Matrix for {icd9_codes[i]}\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(cm_dir / f"confusion_matrix_{icd9_codes[i]}.png")
            plt.close()
        else:
            plt.show()

def get_model_performance_summary(y_true, y_pred, icd9_codes, threshold=0.5):
    """
    Generate a summary of model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        icd9_codes: List of ICD-9 codes
        threshold: Threshold for binary classification
    
    Returns:
        dict: Performance summary
    """
    # Convert to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate overall metrics
    true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
    false_positives = np.sum((y_true == 0) & (y_pred_binary == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred_binary == 0))
    true_negatives = np.sum((y_true == 0) & (y_pred_binary == 0))
    
    # Calculate micro-averaged metrics
    micro_precision = true_positives / max(true_positives + false_positives, 1)
    micro_recall = true_positives / max(true_positives + false_negatives, 1)
    micro_f1 = 2 * (micro_precision * micro_recall) / max(micro_precision + micro_recall, 1e-8)
    
    # Calculate macro-averaged metrics
    macro_precision = np.mean([
        np.sum((y_pred_binary[:, i] == 1) & (y_true[:, i] == 1)) / max(np.sum(y_pred_binary[:, i] == 1), 1)
        for i in range(y_true.shape[1])
    ])
    
    macro_recall = np.mean([
        np.sum((y_pred_binary[:, i] == 1) & (y_true[:, i] == 1)) / max(np.sum(y_true[:, i] == 1), 1)
        for i in range(y_true.shape[1])
    ])
    
    macro_f1 = 2 * (macro_precision * macro_recall) / max(macro_precision + macro_recall, 1e-8)
    
    # Calculate per-code metrics
    code_metrics = []
    for i in range(len(icd9_codes)):
        tp = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
        
        # Calculate AUC
        if len(np.unique(y_true[:, i])) > 1:  # Only calculate if both classes are present
            roc_auc = auc(roc_curve(y_true[:, i], y_pred[:, i])[0], roc_curve(y_true[:, i], y_pred[:, i])[1])
        else:
            roc_auc = 0
        
        code_metrics.append({
            'code': icd9_codes[i],
            'prevalence': float(np.mean(y_true[:, i])),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(roc_auc)
        })
    
    # Create summary
    summary = {
        'overall': {
            'samples': len(y_true),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_negatives': int(true_negatives),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'micro_f1': float(micro_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1)
        },
        'per_code': code_metrics
    }
    
    return summary

def evaluate_model(args):
    """
    Evaluate the trained model on the test set.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ICD-9 codes
    icd9_codes = load_icd9_codes(args.model_dir)
    logger.info(f"Loaded {len(icd9_codes)} ICD-9 codes")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir, len(icd9_codes), device)
    
    # Load test data
    logger.info("Loading test data...")
    
    if args.test_data_path.endswith('.csv'):
        # Load from CSV
        df = pd.read_csv(args.test_data_path)
    else:
        # Load from DuckDB
        conn = duckdb.connect(database=':memory:')
        data_query = f"SELECT * FROM '{args.test_data_path}'"
        
        # Get ICD-9 columns
        icd9_columns = [f"ICD9_{code}" for code in icd9_codes]
        
        # Get the data
        df = duckdb.query(f"""
            SELECT 
                HADM_ID, 
                {', '.join(icd9_columns)}, 
                {args.text_column}
            FROM {data_query}
            LIMIT {args.max_samples if args.max_samples > 0 else 'ALL'}
        """).df()
    
    logger.info(f"Loaded {len(df)} test samples")
    
    # Create dataset
    icd9_columns = [f"ICD9_{code}" for code in icd9_codes]
    
    # Check if all ICD9 columns exist
    missing_columns = [col for col in icd9_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing {len(missing_columns)} ICD-9 columns in test data: {missing_columns}")
        
        # Add missing columns with zeros
        for col in missing_columns:
            df[col] = 0
    
    # Get dummy empty arrays for features not in simplified dataset
    empty_section_features = {}
    empty_binary_indicators = np.zeros((len(df), 0))
    empty_numerical_features = np.zeros((len(df), 0))

    # Create dataset with the enhanced dataset class
    test_dataset = MIMIC_ICD9_Dataset(
        texts=df[args.text_column].values,
        section_features=empty_section_features,
        binary_indicators=empty_binary_indicators,
        numerical_features=empty_numerical_features,
        labels=df[icd9_columns].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )   
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Get predictions
    logger.info("Generating predictions...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with empty section_texts, binary_indicators, and numerical_features
            section_texts = {}
            if hasattr(model, 'section_names') and model.section_names:
                # Create empty section texts dictionary with the correct section names
                for section_name in model.section_names:
                    section_texts[section_name] = {
                        'input_ids': torch.zeros(input_ids.size(0), 128, dtype=torch.long, device=device),
                        'attention_mask': torch.zeros(input_ids.size(0), 128, dtype=torch.long, device=device)
                    }

            # Create empty binary and numerical features with the right dimensions
            if hasattr(model, 'binary_layer') and model.binary_layer is not None:
                bin_in = model.binary_layer[0].in_features
                binary_indicators = torch.zeros(input_ids.size(0), bin_in, device=device)
            else:
                binary_indicators = None
            if hasattr(model, 'numerical_layer') and model.numerical_layer is not None:
                num_in = model.numerical_layer[0].in_features
                numerical_features = torch.zeros(input_ids.size(0), num_in, device=device)
            else:
                numerical_features = None

            # Now call the model with all parameters
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                section_texts=section_texts if section_texts else None,
                binary_indicators=binary_indicators if binary_indicators.size(1) > 0 else None,
                numerical_features=numerical_features if numerical_features.size(1) > 0 else None
            )
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            # Store predictions and labels
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate batch results
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Generate performance summary
    logger.info("Generating performance summary...")
    summary = get_model_performance_summary(all_labels, all_preds, icd9_codes, threshold=args.threshold)
    
    # Save summary
    with open(output_dir / "performance_summary.json", 'w') as f:
        json.dump(summary, f, indent=4, default=float)
    
    # Print overall metrics
    logger.info(f"Overall Performance (threshold={args.threshold}):")
    logger.info(f"Micro-averaged Precision: {summary['overall']['micro_precision']:.4f}")
    logger.info(f"Micro-averaged Recall: {summary['overall']['micro_recall']:.4f}")
    logger.info(f"Micro-averaged F1: {summary['overall']['micro_f1']:.4f}")
    logger.info(f"Macro-averaged Precision: {summary['overall']['macro_precision']:.4f}")
    logger.info(f"Macro-averaged Recall: {summary['overall']['macro_recall']:.4f}")
    logger.info(f"Macro-averaged F1: {summary['overall']['macro_f1']:.4f}")
    
    # Generate code-wise metrics
    code_metrics = pd.DataFrame(summary['per_code'])
    code_metrics = code_metrics.sort_values('prevalence', ascending=False)
    code_metrics.to_csv(output_dir / "code_metrics.csv", index=False)
    
    # Print top 5 codes by prevalence
    logger.info("\nTop 5 codes by prevalence:")
    for _, row in code_metrics.head(5).iterrows():
        logger.info(f"{row['code']}: Prevalence={row['prevalence']:.4f}, Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, F1={row['f1']:.4f}, AUC={row['auc']:.4f}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # ROC curves
    plot_roc_curves(all_labels, all_preds, icd9_codes, top_n=args.top_n, save_path=output_dir)
    
    # Precision-recall curves
    plot_precision_recall_curves(all_labels, all_preds, icd9_codes, top_n=args.top_n, save_path=output_dir)
    
    # Threshold analysis
    plot_threshold_analysis(all_labels, all_preds, save_path=output_dir)
    
    # Confusion matrices
    plot_confusion_matrix(all_labels, all_preds, icd9_codes, top_n=args.top_n, threshold=args.threshold, save_path=output_dir)
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained BERT-CNN model for ICD-9 code prediction")
    
    # Data arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save the evaluation results")
    parser.add_argument("--text_column", type=str, default="clinical_weighted_text", 
                      help="Column to use for text (clinical_weighted_text or summary_snippet_clean)")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for BERT")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top codes to show in visualizations")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to evaluate (-1 for all)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)