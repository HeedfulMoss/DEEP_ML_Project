import os
import sys
import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
import logging
import argparse
from tqdm import tqdm
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("icd9_bert_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MIMIC_ICD9_Enhanced_Dataset(Dataset):
    """
    Enhanced PyTorch Dataset for MIMIC ICD-9 code prediction that includes additional features.
    """
    def __init__(self, texts, section_features, binary_indicators, numerical_features, labels, tokenizer, max_length=512):
        """
        Initialize the enhanced dataset.
        
        Args:
            texts: List of main clinical texts
            section_features: Dict of section texts (key: section name, value: list of texts)
            binary_indicators: Array of binary section presence indicators
            numerical_features: Array of numerical features (lengths and word counts)
            labels: Array of one-hot encoded ICD-9 codes
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length for BERT tokenizer
        """
        self.texts = texts
        self.section_features = section_features
        self.binary_indicators = binary_indicators
        self.numerical_features = numerical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Encode main text
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get input IDs and attention mask for main text
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Get section texts (encoded on the fly to save memory)
        section_texts = {}
        for section_name, texts in self.section_features.items():
            if texts[idx]:  # If section text exists
                # Encode with a shorter max length for sections
                section_encoding = self.tokenizer(
                    str(texts[idx]),
                    truncation=True,
                    padding='max_length',
                    max_length=128,  # Shorter for sections
                    return_tensors='pt'
                )
                section_texts[section_name] = {
                    'input_ids': section_encoding['input_ids'].squeeze(0),
                    'attention_mask': section_encoding['attention_mask'].squeeze(0)
                }
            else:
                # If section doesn't exist, use empty tensors
                section_texts[section_name] = {
                    'input_ids': torch.zeros(128, dtype=torch.long),
                    'attention_mask': torch.zeros(128, dtype=torch.long)
                }
        
        # Get binary indicators for this sample
        binary_indicators = torch.tensor(self.binary_indicators[idx], dtype=torch.float)
        
        # Get numerical features for this sample
        numerical_features = torch.tensor(self.numerical_features[idx], dtype=torch.float)
        
        # Get labels for this sample
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'section_texts': section_texts,
            'binary_indicators': binary_indicators,
            'numerical_features': numerical_features,
            'labels': labels
        }

class Enhanced_BERT_CNN_ICD9(nn.Module):
    """
    Enhanced model combining BERT embeddings with CNN for the main text,
    separate processing for section texts, and integration of binary indicators
    and numerical features for ICD-9 code prediction.
    """
    def __init__(self, num_labels=20, bert_model_name='bert-base-uncased', dropout_rate=0.1,
                 section_names=None, num_binary_features=0, num_numerical_features=0):
        """
        Initialize the enhanced model.
        
        Args:
            num_labels: Number of ICD-9 codes to predict (default: 20)
            bert_model_name: Pretrained BERT model to use
            dropout_rate: Dropout rate for regularization
            section_names: List of section names
            num_binary_features: Number of binary features (has_X)
            num_numerical_features: Number of numerical features (lengths and word counts)
        """
        super(Enhanced_BERT_CNN_ICD9, self).__init__()
        
        self.section_names = section_names if section_names else []
        
        # Initialize BERT model for main text
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Get the hidden size from BERT config
        hidden_size = self.bert.config.hidden_size  # Usually 768 for BERT base
        
        # CNN layers for main text with different kernel sizes
        self.conv1 = nn.Conv1d(hidden_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, 128, kernel_size=5, padding=2)
        
        # Section text processing
        self.section_encoders = nn.ModuleDict()
        self.section_convs = nn.ModuleDict()
        
        for section_name in self.section_names:
            # Use lightweight CNN for each section (not full BERT to save memory)
            self.section_convs[section_name] = nn.Conv1d(hidden_size, 64, kernel_size=3, padding=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the total feature size
        main_text_features = 128 * 3  # From 3 CNN layers
        section_features = 64 * len(self.section_names)  # 64 per section
        
        # Process binary indicators (has_X)
        self.binary_layer = None
        if num_binary_features > 0:
            self.binary_layer = nn.Sequential(
                nn.Linear(num_binary_features, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            binary_features = 32
        else:
            binary_features = 0
        
        # Process numerical features (section_X_length, section_X_words)
        self.numerical_layer = None
        if num_numerical_features > 0:
            self.numerical_layer = nn.Sequential(
                nn.Linear(num_numerical_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            numerical_features = 64
        else:
            numerical_features = 0
        
        # Calculate total feature size
        total_features = main_text_features + section_features + binary_features + numerical_features
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer for multi-label classification
        self.classifier = nn.Linear(256, num_labels)
    
    def forward(self, input_ids, attention_mask, section_texts=None, binary_indicators=None, numerical_features=None):
        """
        Forward pass of the enhanced model.
        
        Args:
            input_ids: Token IDs from BERT tokenizer for main text
            attention_mask: Attention mask from BERT tokenizer for main text
            section_texts: Dict of section texts with their input_ids and attention_masks
            binary_indicators: Binary indicators for section presence
            numerical_features: Numerical features for section lengths and word counts
        
        Returns:
            logits: Raw prediction scores for each ICD-9 code
        """
        # Process main text with BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        main_embeddings = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        
        # Process main text with CNN layers
        x = main_embeddings.transpose(1, 2)  # [batch_size, hidden_size, seq_length]
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        # Apply global max pooling to main text features
        x1 = torch.max(x1, dim=2)[0]  # [batch_size, 128]
        x2 = torch.max(x2, dim=2)[0]  # [batch_size, 128]
        x3 = torch.max(x3, dim=2)[0]  # [batch_size, 128]
        
        # Concatenate main text CNN features
        main_features = torch.cat((x1, x2, x3), dim=1)  # [batch_size, 384]
        
        # Process section texts if provided
        section_features_list = []
        
        if section_texts is not None:
            # Process each section text with the BERT model and its CNN
            for section_name in self.section_names:
                # Check if the section exists in the batch
                if section_name in section_texts:
                    # Get section text features
                    section_input_ids = section_texts[section_name]['input_ids']
                    section_attention_mask = section_texts[section_name]['attention_mask']
                    
                    # Skip BERT encoding for empty sections (all zeros)
                    if torch.sum(section_attention_mask) == 0:
                        # Use zero vector for empty sections
                        section_feature = torch.zeros(input_ids.size(0), 64, device=input_ids.device)
                    else:
                        # Process with BERT and CNN
                        with torch.no_grad():  # We don't need gradients for section texts
                            section_output = self.bert(input_ids=section_input_ids, attention_mask=section_attention_mask)
                        section_embeddings = section_output.last_hidden_state
                        section_embeddings = section_embeddings.transpose(1, 2)  # [batch_size, hidden_size, seq_length]
                        section_conv = self.section_convs[section_name](section_embeddings)
                        section_feature = torch.max(F.relu(section_conv), dim=2)[0]  # [batch_size, 64]
                    
                    section_features_list.append(section_feature)
        
        # Concatenate all section features if any
        if section_features_list:
            all_section_features = torch.cat(section_features_list, dim=1)
        else:
            all_section_features = torch.zeros(input_ids.size(0), 0, device=input_ids.device)
        
        # Process binary indicators if provided
        if binary_indicators is not None and self.binary_layer is not None:
            binary_features = self.binary_layer(binary_indicators)
        else:
            binary_features = torch.zeros(input_ids.size(0), 0, device=input_ids.device)
        
        # Process numerical features if provided
        if numerical_features is not None and self.numerical_layer is not None:
            numerical_features_processed = self.numerical_layer(numerical_features)
        else:
            numerical_features_processed = torch.zeros(input_ids.size(0), 0, device=input_ids.device)
        
        # Concatenate all features
        all_features = []
        if main_features.size(1) > 0:
            all_features.append(main_features)
        if all_section_features.size(1) > 0:
            all_features.append(all_section_features)
        if binary_features.size(1) > 0:
            all_features.append(binary_features)
        if numerical_features_processed.size(1) > 0:
            all_features.append(numerical_features_processed)
        
        combined_features = torch.cat(all_features, dim=1)
        
        # Apply feature fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Apply dropout for regularization
        fused_features = self.dropout(fused_features)
        
        # Apply final classification layer
        logits = self.classifier(fused_features)
        
        return logits

def verify_dataset_format(full_data_query, template_data_path):
    """
    Verify that the full dataset has the same structure as the template dataset.
    
    Args:
        full_data_query: DuckDB query for the full dataset
        template_data_path: Path to the template dataset
    
    Returns:
        bool: True if the format matches, False otherwise
    """
    # Get column names from full dataset
    full_columns = duckdb.query(f"SELECT * FROM {full_data_query} LIMIT 0").columns
    
    # Get column names from template dataset
    template_df = pd.read_csv(template_data_path)
    template_columns = template_df.columns.tolist()
    
    # Check if columns match
    columns_match = set(full_columns) == set(template_columns)
    if not columns_match:
        full_set = set(full_columns)
        template_set = set(template_columns)
        missing_in_full = template_set - full_set
        extra_in_full = full_set - template_set
        
        logger.warning(f"Column mismatch! Missing in full: {missing_in_full}, Extra in full: {extra_in_full}")
        return False
    
    # Check a sample of data types
    sample_full = duckdb.query(f"SELECT * FROM {full_data_query} LIMIT 20").df()
    
    # Only check columns that exist in both datasets
    common_columns = set(sample_full.columns).intersection(set(template_df.columns))
    
    for col in common_columns:
        # Skip checking data types for text columns which might vary
        if "section_" in col or "_text" in col or "snippet" in col:
            continue
            
        # For numeric columns, ensure they're the same type or can be converted
        if pd.api.types.is_numeric_dtype(template_df[col]):
            if not pd.api.types.is_numeric_dtype(sample_full[col]):
                logger.warning(f"Column {col} has different data types: {template_df[col].dtype} vs {sample_full[col].dtype}")
                return False
    
    return True

def load_data(preprocessed_data_path, template_data_path):
    """
    Load the preprocessed dataset using DuckDB and verify its format.
    
    Args:
        preprocessed_data_path: Path to the preprocessed dataset
        template_data_path: Path to the template dataset for verification
    
    Returns:
        DuckDB connection and query for the dataset
    """
    try:
        # Connect to DuckDB
        conn = duckdb.connect(database=':memory:')
        
        # Check if the file exists
        if not os.path.exists(preprocessed_data_path):
            logger.error(f"File not found: {preprocessed_data_path}")
            raise FileNotFoundError(f"File not found: {preprocessed_data_path}")
        
        # Create a query for the data
        data_query = f"read_csv_auto('{preprocessed_data_path}')"

        
        # Verify format against template
        is_valid = verify_dataset_format(data_query, template_data_path)
        if not is_valid:
            logger.warning("Dataset format validation failed. Proceeding with caution.")
        else:
            logger.info("Dataset format validation passed.")
        
        # Get dataset statistics
        row_count = duckdb.query(f"SELECT COUNT(*) FROM {data_query}").fetchone()[0]
        logger.info(f"Dataset loaded with {row_count} rows.")

        return conn, data_query
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def collate_enhanced_batch(batch):
    """
    Custom collate function for the enhanced dataset.
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        dict: Collated batch
    """
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    # Initialize result dict
    result = {}
    
    # Collate input_ids and attention_mask
    for key in ['input_ids', 'attention_mask', 'binary_indicators', 'numerical_features', 'labels']:
        if key in keys:
            result[key] = torch.stack([sample[key] for sample in batch])
    
    # Collate section_texts
    if 'section_texts' in keys:
        section_texts = {}
        section_names = batch[0]['section_texts'].keys()
        
        for section_name in section_names:
            section_texts[section_name] = {
                'input_ids': torch.stack([sample['section_texts'][section_name]['input_ids'] for sample in batch]),
                'attention_mask': torch.stack([sample['section_texts'][section_name]['attention_mask'] for sample in batch])
            }
        
        result['section_texts'] = section_texts
    
    return result

def train_epoch(model, data_loader, optimizer, criterion, device, scheduler=None):
    """
    Train the enhanced model for one epoch.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on (CPU or GPU)
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Move section_texts to device if present
        section_texts = None
        if 'section_texts' in batch:
            section_texts = {}
            for section_name, section_data in batch['section_texts'].items():
                section_texts[section_name] = {
                    'input_ids': section_data['input_ids'].to(device),
                    'attention_mask': section_data['attention_mask'].to(device)
                }
        
        # Move binary_indicators to device if present
        binary_indicators = None
        if 'binary_indicators' in batch:
            binary_indicators = batch['binary_indicators'].to(device)
        
        # Move numerical_features to device if present
        numerical_features = None
        if 'numerical_features' in batch:
            numerical_features = batch['numerical_features'].to(device)
        
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            section_texts=section_texts,
            binary_indicators=binary_indicators,
            numerical_features=numerical_features
        )
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, criterion, device, threshold=0.5):
    """
    Evaluate the enhanced model on validation or test data.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on (CPU or GPU)
        threshold: Threshold for binary classification
    
    Returns:
        tuple: Dictionary of metrics, predictions, and labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Move section_texts to device if present
            section_texts = None
            if 'section_texts' in batch:
                section_texts = {}
                for section_name, section_data in batch['section_texts'].items():
                    section_texts[section_name] = {
                        'input_ids': section_data['input_ids'].to(device),
                        'attention_mask': section_data['attention_mask'].to(device)
                    }
            
            # Move binary_indicators to device if present
            binary_indicators = None
            if 'binary_indicators' in batch:
                binary_indicators = batch['binary_indicators'].to(device)
            
            # Move numerical_features to device if present
            numerical_features = None
            if 'numerical_features' in batch:
                numerical_features = batch['numerical_features'].to(device)
            
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                section_texts=section_texts,
                binary_indicators=binary_indicators,
                numerical_features=numerical_features
            )
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            # Store predictions and labels
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate batch results
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    
    # Calculate AUC-ROC (area under the ROC curve)
    auc_scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:  # Only calculate if both classes are present
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
    
    avg_auc = np.mean(auc_scores) if auc_scores else 0
    
    # Convert to binary predictions using threshold
    binary_preds = (all_preds >= threshold).astype(int)
    
    # Calculate precision, recall, F1-score (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, binary_preds, average='macro', zero_division=0
    )
    
    # Create metrics dictionary
    metrics = {
        'loss': avg_loss,
        'auc': avg_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, all_preds, all_labels

def prepare_enhanced_dataset(df, icd9_columns, main_text_column, section_names, binary_columns, numerical_columns):
    """
    Prepare the enhanced dataset from a DataFrame.
    
    Args:
        df: DataFrame containing the data
        icd9_columns: List of ICD-9 code columns
        main_text_column: Column name for the main clinical text
        section_names: List of section names
        binary_columns: List of binary indicator columns
        numerical_columns: List of numerical feature columns
    
    Returns:
        tuple: Lists of features for the enhanced dataset
    """
    # Main text
    texts = df[main_text_column].values
    
    # Section texts
    section_features = {}
    for section in section_names:
        section_col = f"section_{section}"
        if section_col in df.columns:
            # Fill NaN with empty string
            section_features[section] = df[section_col].fillna('').values
    
    # Binary indicators
    if binary_columns:
        binary_indicators = df[binary_columns].values
    else:
        binary_indicators = np.zeros((len(df), 0))
    
    # Numerical features
    if numerical_columns:
        # Fill NaN with 0
        numerical_features = df[numerical_columns].fillna(0).values
    else:
        numerical_features = np.zeros((len(df), 0))
    
    # Labels
    labels = df[icd9_columns].values
    
    return texts, section_features, binary_indicators, numerical_features, labels

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    icd9_codes,
    epochs=5, 
    early_stopping_patience=3, 
    scheduler=None,
    save_dir=None
):
    """
    Train and validate the enhanced model.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to train on (CPU or GPU)
        icd9_codes: List of ICD-9 codes for metrics
        epochs: Number of training epochs
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        scheduler: Learning rate scheduler (optional)
        save_dir: Directory to save model checkpoints
    
    Returns:
        dict: Training history and best metrics
    """
    # Create timestamp for unique run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bert_cnn_icd9_{timestamp}"
    
    # Create save directory if specified
    if save_dir:
        save_path = Path(save_dir) / run_name
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model checkpoints and results will be saved to {save_path}")
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # Initialize early stopping variables
    best_val_auc = 0
    best_model_state = None
    best_epoch = 0
    no_improvement_count = 0
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        
        # Evaluate on validation set
        val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Check if this is the best model so far (based on AUC)
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            no_improvement_count = 0
            
            # Save best model if save_dir is specified
            if save_dir:
                torch.save(model.state_dict(), save_path / "best_model.pt")
                
                # Save predictions for best model
                np.save(save_path / "best_val_preds.npy", val_preds)
                np.save(save_path / "best_val_labels.npy", val_labels)
                
                # Save metrics for best model
                with open(save_path / "best_metrics.json", 'w') as f:
                    json.dump({
                        'epoch': best_epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_metrics['loss'],
                        'val_auc': val_metrics['auc'],
                        'val_precision': val_metrics['precision'],
                        'val_recall': val_metrics['recall'],
                        'val_f1': val_metrics['f1']
                    }, f, indent=4)
        else:
            no_improvement_count += 1
        
        # Save checkpoint every epoch if save_dir is specified
        if save_dir:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics
            }, save_path / f"checkpoint_epoch_{epoch+1}.pt")
            
            # Save history after each epoch
            with open(save_path / "training_history.json", 'w') as f:
                json.dump(history, f, indent=4)
        
        # Early stopping
        if no_improvement_count >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model from epoch {best_epoch+1}")
    
    # Return the training history and best metrics
    return {
        'history': history,
        'best_epoch': best_epoch + 1,
        'best_val_auc': best_val_auc,
        'run_name': run_name
    }

def save_model_info(model, section_names, binary_columns, numerical_columns, save_path):
    """
    Save model configuration and feature information.
    
    Args:
        model: The neural network model
        section_names: List of section names
        binary_columns: List of binary indicator columns
        numerical_columns: List of numerical feature columns
        save_path: Path to save the information
    """
    model_info = {
        'section_names': section_names,
        'binary_columns': binary_columns,
        'numerical_columns': numerical_columns,
        'model_parameters': {
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    with open(save_path / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=4)

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train an enhanced BERT-CNN model for ICD-9 code prediction")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory containing the data")
    parser.add_argument("--output_dir", type=str, default="../models/bert_icd9", help="Directory to save the model and results")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for BERT")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for regularization")
    
    # Feature selection arguments
    parser.add_argument("--use_section_texts", action="store_true", help="Whether to use section texts as features")
    parser.add_argument("--use_binary_indicators", action="store_true", help="Whether to use binary indicators as features")
    parser.add_argument("--use_numerical_features", action="store_true", help="Whether to use numerical features as features")
    parser.add_argument("--text_column", type=str, default="summary_snippet_clean", 
                      help="Column to use for main text (clinical_weighted_text or summary_snippet_clean)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def get_dataset_feature_columns(df):
    """
    Get feature columns from the dataset.
    
    Args:
        df: DataFrame containing the data
    
    Returns:
        tuple: Lists of section names, binary columns, and numerical columns
    """
    # Find section text columns (section_X)
    section_columns = [col for col in df.columns if col.startswith("section_") and not (col.endswith("_length") or col.endswith("_words"))]
    section_names = [col.replace("section_", "") for col in section_columns]
    
    # Find binary indicator columns (has_X)
    binary_columns = [col for col in df.columns if col.startswith("has_")]
    
    # Find numerical feature columns (section_X_length, section_X_words)
    length_columns = [col for col in df.columns if col.endswith("_length")]
    word_columns = [col for col in df.columns if col.endswith("_words")]
    numerical_columns = length_columns + word_columns
    
    return section_names, binary_columns, numerical_columns

def main():
    """
    Main function to train and evaluate the enhanced model.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Save run configuration
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Load data
    logger.info("Loading data...")
    preprocessed_data_path = Path(args.data_dir) / "preprocessed" / "summary_results.csv"
    template_data_path = Path(args.data_dir) / "preprocessed" / "summary_results_trimmed.csv"
    
    conn, data_query = load_data(preprocessed_data_path, template_data_path)
    
    # Get ICD-9 codes (column names starting with "ICD9_")
    icd9_columns = [col for col in duckdb.query(f"SELECT * FROM {data_query} LIMIT 0").columns if col.startswith("ICD9_")]
    icd9_codes = [col[5:] for col in icd9_columns]  # Remove "ICD9_" prefix
    
    logger.info(f"Found {len(icd9_codes)} ICD-9 codes: {', '.join(icd9_codes)}")
    
    # Load a sample of data to get feature columns
    sample_df = duckdb.query(f"SELECT * FROM {data_query} LIMIT 100").df()
    section_names, binary_columns, numerical_columns = get_dataset_feature_columns(sample_df)
    
    logger.info(f"Found {len(section_names)} section text columns")
    logger.info(f"Found {len(binary_columns)} binary indicator columns")
    logger.info(f"Found {len(numerical_columns)} numerical feature columns")
    
    # Filter features based on arguments
    if not args.use_section_texts:
        section_names = []
        logger.info("Section texts will not be used")
    
    if not args.use_binary_indicators:
        binary_columns = []
        logger.info("Binary indicators will not be used")
    
    if not args.use_numerical_features:
        numerical_columns = []
        logger.info("Numerical features will not be used")
    
    # Save feature information
    with open(run_dir / "feature_info.json", "w") as f:
        json.dump({
            'section_names': section_names,
            'binary_columns': binary_columns,
            'numerical_columns': numerical_columns,
            'icd9_codes': icd9_codes,
            'main_text_column': args.text_column
        }, f, indent=4)
    
    # Save the ICD-9 label binarizer
    label_binarizer = {
        'index_to_code': {i: code for i, code in enumerate(icd9_codes)},
        'code_to_index': {code: i for i, code in enumerate(icd9_codes)}
    }
    with open(output_dir / "icd9_label_binarizer.pkl", 'wb') as f:
        pickle.dump(label_binarizer, f)
    
    # Get data from DuckDB
    logger.info("Preparing data for training...")
    
    # Construct the SQL query based on the selected features
    select_columns = ["HADM_ID", *icd9_columns, args.text_column]
    
    # Add section text columns if used
    if args.use_section_texts:
        select_columns.extend([f"section_{section}" for section in section_names])
    
    # Add binary indicator columns if used
    if args.use_binary_indicators:
        select_columns.extend(binary_columns)
    
    # Add numerical feature columns if used
    if args.use_numerical_features:
        select_columns.extend(numerical_columns)
    
    # Execute the query
    df = duckdb.query(f"""
        SELECT {', '.join(select_columns)}
        FROM {data_query}
    """).df()
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)
    
    logger.info(f"Data split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Prepare datasets
    logger.info("Preparing enhanced datasets...")
    
    # Prepare train dataset
    train_texts, train_section_features, train_binary_indicators, train_numerical_features, train_labels = prepare_enhanced_dataset(
        train_df, icd9_columns, args.text_column, section_names, binary_columns, numerical_columns
    )
    
    train_dataset = MIMIC_ICD9_Enhanced_Dataset(
        texts=train_texts,
        section_features=train_section_features,
        binary_indicators=train_binary_indicators,
        numerical_features=train_numerical_features,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Prepare validation dataset
    val_texts, val_section_features, val_binary_indicators, val_numerical_features, val_labels = prepare_enhanced_dataset(
        val_df, icd9_columns, args.text_column, section_names, binary_columns, numerical_columns
    )
    
    val_dataset = MIMIC_ICD9_Enhanced_Dataset(
        texts=val_texts,
        section_features=val_section_features,
        binary_indicators=val_binary_indicators,
        numerical_features=val_numerical_features,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Prepare test dataset
    test_texts, test_section_features, test_binary_indicators, test_numerical_features, test_labels = prepare_enhanced_dataset(
        test_df, icd9_columns, args.text_column, section_names, binary_columns, numerical_columns
    )
    
    test_dataset = MIMIC_ICD9_Enhanced_Dataset(
        texts=test_texts,
        section_features=test_section_features,
        binary_indicators=test_binary_indicators,
        numerical_features=test_numerical_features,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_enhanced_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_enhanced_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_enhanced_batch
    )
    
    # Initialize model
    logger.info("Initializing enhanced model...")
    model = Enhanced_BERT_CNN_ICD9(
        num_labels=len(icd9_codes),
        bert_model_name=args.model_name,
        dropout_rate=args.dropout_rate,
        section_names=section_names,
        num_binary_features=len(binary_columns),
        num_numerical_features=len(numerical_columns)
    )
    model.to(device)
    
    # Save model info
    save_model_info(model, section_names, binary_columns, numerical_columns, run_dir)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Initialize loss function for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        icd9_codes=icd9_codes,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        scheduler=scheduler,
        save_dir=run_dir
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    
    # Save test metrics
    with open(run_dir / "test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Save tokenizer
    tokenizer.save_pretrained(run_dir / "tokenizer")
    
    # Create feature contribution analysis
    if args.use_section_texts or args.use_binary_indicators or args.use_numerical_features:
        logger.info("Analyzing feature contributions...")
        
        feature_groups = []
        if args.use_section_texts:
            feature_groups.append("section_texts")
        if args.use_binary_indicators:
            feature_groups.append("binary_indicators")
        if args.use_numerical_features:
            feature_groups.append("numerical_features")
        
        feature_contributions = {
            'baseline': test_metrics['auc'],
            'feature_groups': {}
        }
        
        # Save test predictions and labels for later analysis
        np.save(run_dir / "test_preds.npy", test_preds)
        np.save(run_dir / "test_labels.npy", test_labels)
        
        # Save feature contribution analysis
        with open(run_dir / "feature_contributions.json", 'w') as f:
            json.dump(feature_contributions, f, indent=4)
    
    # Save best model to output directory
    logger.info(f"Saving best model to {output_dir}")
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    
    logger.info(f"Training and evaluation completed. Results saved to {run_dir}")
    
    return run_dir

if __name__ == "__main__":
    main()
