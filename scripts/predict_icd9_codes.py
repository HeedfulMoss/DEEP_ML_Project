import os
import sys
import json
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer
import argparse
import logging
import csv
from tqdm import tqdm

# Import model class from training script
from train_icd9_predictor import BERT_CNN_ICD9, predict_icd9_codes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("icd9_prediction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_dir, device):
    """
    Load the trained model, tokenizer, and ICD-9 codes.
    
    Args:
        model_dir: Directory containing the model files
        device: Device to load the model on
    
    Returns:
        tuple: Loaded model, tokenizer, and ICD-9 codes
    """
    # Load ICD-9 codes
    label_binarizer_path = Path(model_dir) / "icd9_label_binarizer.pkl"
    
    if not label_binarizer_path.exists():
        # Check parent directory
        label_binarizer_path = Path(model_dir).parent / "icd9_label_binarizer.pkl"
    
    if not label_binarizer_path.exists():
        # Try to find in subfolders
        for subfolder in os.listdir(model_dir):
            subfolder_path = Path(model_dir) / subfolder
            if subfolder_path.is_dir():
                potential_path = subfolder_path / "icd9_label_binarizer.pkl"
                if potential_path.exists():
                    label_binarizer_path = potential_path
                    break
    
    if not label_binarizer_path.exists():
        raise FileNotFoundError(f"No ICD-9 label binarizer found in {model_dir}")
    
    with open(label_binarizer_path, 'rb') as f:
        label_binarizer = pickle.load(f)
    
    icd9_codes = [label_binarizer['index_to_code'][i] for i in range(len(label_binarizer['index_to_code']))]
    logger.info(f"Loaded {len(icd9_codes)} ICD-9 codes")
    
    # Find model file
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
    
    # Initialize model
    model = BERT_CNN_ICD9(num_labels=len(icd9_codes))
    
    # Load model weights
    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, icd9_codes

def process_file(file_path, model, tokenizer, icd9_codes, device, threshold=0.5, output_format="json"):
    """
    Process a single file and predict ICD-9 codes.
    
    Args:
        file_path: Path to the text file
        model: The trained model
        tokenizer: The BERT tokenizer
        icd9_codes: List of ICD-9 codes
        device: Device to run inference on
        threshold: Threshold for binary classification
        output_format: Format of the output ("json" or "csv")
    
    Returns:
        dict: Prediction results
    """
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Predict
    result = predict_icd9_codes(text, model, tokenizer, icd9_codes, device, threshold=threshold)
    
    # Add file name
    result['file_name'] = file_path.name
    
    return result

def process_csv(csv_path, model, tokenizer, icd9_codes, device, text_column, threshold=0.5, output_dir=None, batch_size=1):
    """
    Process a CSV file with multiple clinical texts and predict ICD-9 codes.
    
    Args:
        csv_path: Path to the CSV file
        model: The trained model
        tokenizer: The BERT tokenizer
        icd9_codes: List of ICD-9 codes
        device: Device to run inference on
        text_column: Name of the column containing clinical text
        threshold: Threshold for binary classification
        output_dir: Directory to save the results
        batch_size: Number of samples to process at once
    
    Returns:
        DataFrame: Prediction results
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV file")
    
    # Prepare output
    output_rows = []
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Predicting"):
        batch_df = df.iloc[i:i+batch_size]
        
        # Process each row in the batch
        for _, row in batch_df.iterrows():
            text = row[text_column]
            
            # Skip empty texts
            if pd.isna(text) or text == "":
                continue
            
            # Predict
            result = predict_icd9_codes(str(text), model, tokenizer, icd9_codes, device, threshold=threshold)
            
            # Create output row
            output_row = {
                'hadm_id': row.get('HADM_ID', '') if 'HADM_ID' in row else '',
                'subject_id': row.get('SUBJECT_ID', '') if 'SUBJECT_ID' in row else '',
                'text_length': len(str(text))
            }
            
            # Add predicted codes
            for code_info in result['predicted_codes']:
                output_row[f"ICD9_{code_info['code']}"] = code_info['probability']
            
            output_rows.append(output_row)
    
    # Convert to DataFrame
    output_df = pd.DataFrame(output_rows)
    
    # Save results if output_dir is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get CSV file name
        csv_name = Path(csv_path).stem
        
        # Save predictions
        output_df.to_csv(output_dir / f"{csv_name}_predictions.csv", index=False)
        
        # Save full probabilities
        if not output_rows:
            logger.warning("No valid predictions to save")
        else:
            # Create a DataFrame with all probabilities
            all_probs_df = pd.DataFrame(
                {f"ICD9_{code}": [row.get(f"ICD9_{code}", 0) for row in output_rows] for code in icd9_codes}
            )
            
            # Add identifiers
            if 'hadm_id' in output_df.columns:
                all_probs_df['HADM_ID'] = output_df['hadm_id']
            if 'subject_id' in output_df.columns:
                all_probs_df['SUBJECT_ID'] = output_df['subject_id']
            
            # Save
            all_probs_df.to_csv(output_dir / f"{csv_name}_all_probabilities.csv", index=False)
    
    return output_df

def interactive_prediction(model, tokenizer, icd9_codes, device, threshold=0.5):
    """
    Interactive mode for predicting ICD-9 codes from user input.
    
    Args:
        model: The trained model
        tokenizer: The BERT tokenizer
        icd9_codes: List of ICD-9 codes
        device: Device to run inference on
        threshold: Threshold for binary classification
    """
    print("\n===== Interactive ICD-9 Code Prediction =====")
    print("Enter clinical text to predict ICD-9 codes. Type 'exit' to quit.")
    
    while True:
        print("\nEnter clinical text:")
        lines = []
        
        while True:
            line = input()
            if line.strip().lower() == "done":
                break
            lines.append(line)
        
        text = "\n".join(lines)
        
        if text.strip().lower() == "exit":
            break
        
        # Predict
        result = predict_icd9_codes(text, model, tokenizer, icd9_codes, device, threshold=threshold)
        
        # Print results
        if not result['predicted_codes']:
            print("No ICD-9 codes predicted with confidence above threshold.")
        else:
            print("\nPredicted ICD-9 codes:")
            for code_info in result['predicted_codes']:
                print(f"- {code_info['code']}: {code_info['probability']:.4f}")
    
    print("Exiting interactive mode.")

def main():
    """
    Main function to predict ICD-9 codes from clinical text.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Predict ICD-9 codes from clinical text")
    
    # Model arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model")
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", type=str, help="Path to a text file containing clinical text")
    input_group.add_argument("--csv", type=str, help="Path to a CSV file containing clinical texts")
    input_group.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    # CSV specific arguments
    parser.add_argument("--text_column", type=str, default="clinical_weighted_text", 
                      help="Column name containing clinical text in the CSV file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing CSV file")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save the predictions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--output_format", type=str, default="json", choices=["json", "csv"], 
                      help="Format of the output for single file prediction")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model, tokenizer, and ICD-9 codes
    model, tokenizer, icd9_codes = load_model_and_tokenizer(args.model_dir, device)
    
    # Process input
    if args.interactive:
        # Interactive mode
        interactive_prediction(model, tokenizer, icd9_codes, device, threshold=args.threshold)
    
    elif args.file:
        # Process single file
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
        
        logger.info(f"Processing file: {file_path}")
        result = process_file(file_path, model, tokenizer, icd9_codes, device, 
                             threshold=args.threshold, output_format=args.output_format)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result
        output_path = output_dir / f"{file_path.stem}_prediction"
        if args.output_format == "json":
            with open(f"{output_path}.json", 'w') as f:
                json.dump(result, f, indent=4)
        else:
            with open(f"{output_path}.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(["code", "probability"])
                
                # Write predicted codes
                for code_info in result['predicted_codes']:
                    writer.writerow([code_info['code'], code_info['probability']])
        
        logger.info(f"Prediction saved to {output_path}.{args.output_format}")
        
        # Print result
        print("\nPredicted ICD-9 codes:")
        for code_info in result['predicted_codes']:
            print(f"- {code_info['code']}: {code_info['probability']:.4f}")
    
    elif args.csv:
        # Process CSV file
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return
        
        logger.info(f"Processing CSV file: {csv_path}")
        output_df = process_csv(csv_path, model, tokenizer, icd9_codes, device, 
                               args.text_column, threshold=args.threshold, 
                               output_dir=args.output_dir, batch_size=args.batch_size)
        
        logger.info(f"Processed {len(output_df)} samples from CSV file")
        logger.info(f"Predictions saved to {args.output_dir}")
    
    logger.info("Prediction completed")

if __name__ == "__main__":
    main()