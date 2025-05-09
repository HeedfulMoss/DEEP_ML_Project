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
import scispacy
import spacy
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
import logging
import argparse
from tqdm import tqdm
import pickle
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ultra_enhanced_icd9_bert_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MIMIC_ICD9_Ultra_Enhanced_Dataset(Dataset):
    """
    Enhanced PyTorch Dataset for MIMIC ICD-9 code prediction that includes additional features.
    """
    def __init__(self, texts, section_features, binary_indicators, numerical_features, 
                medical_concepts=None, labels=None, tokenizer=None, max_length=512, is_test=False):
        """
        Initialize the enhanced dataset.
        
        Args:
            texts: List of main clinical texts
            section_features: Dict of section texts (key: section name, value: list of texts)
            binary_indicators: Array of binary section presence indicators
            numerical_features: Array of numerical features (lengths and word counts)
            medical_concepts: Dict of extracted medical concepts (optional)
            labels: Array of one-hot encoded ICD-9 codes (optional for test data)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length for BERT tokenizer
            is_test: Flag to indicate if this is test data (no labels)
        """
        self.texts = texts
        self.section_features = section_features
        self.binary_indicators = binary_indicators
        self.numerical_features = numerical_features
        self.medical_concepts = medical_concepts if medical_concepts is not None else {}
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
    
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
        
        # Get section texts
        section_texts = {}
        for section_name, texts in self.section_features.items():
            if idx < len(texts) and texts[idx]:  # If section text exists
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
        
        # Get medical concept features if available
        medical_concept_features = torch.zeros(1, dtype=torch.float)  # Default empty
        if self.medical_concepts and len(self.medical_concepts) > 0 and idx in self.medical_concepts:
            medical_concept_features = torch.tensor(self.medical_concepts[idx], dtype=torch.float)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'section_texts': section_texts,
            'binary_indicators': binary_indicators,
            'numerical_features': numerical_features,
            'medical_concept_features': medical_concept_features
        }
        
        # Add labels if this is not test data
        if not self.is_test and self.labels is not None:
            result['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return result

class Ultra_Enhanced_BERT_CNN_ICD9(nn.Module):
    """
    Enhanced model combining BERT embeddings with CNN for the main text,
    separate processing for section texts, and integration of binary indicators,
    numerical features, and medical concepts for ICD-9 code prediction.
    """
    def __init__(self, num_labels=20, bert_model_name='bert-base-uncased', dropout_rate=0.1,
                 section_names=None, num_binary_features=0, num_numerical_features=0,
                 num_medical_concept_features=0):
        """
        Initialize the enhanced model.
        
        Args:
            num_labels: Number of ICD-9 codes to predict (default: 20)
            bert_model_name: Pretrained BERT model to use
            dropout_rate: Dropout rate for regularization
            section_names: List of section names
            num_binary_features: Number of binary features (has_X)
            num_numerical_features: Number of numerical features (lengths and word counts)
            num_medical_concept_features: Number of medical concept features
        """
        super(Ultra_Enhanced_BERT_CNN_ICD9, self).__init__()
        
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
            
        # Process medical concept features
        self.medical_concept_layer = None
        if num_medical_concept_features > 0:
            self.medical_concept_layer = nn.Sequential(
                nn.Linear(num_medical_concept_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            medical_concept_features = 64
        else:
            medical_concept_features = 0
        
        # Calculate total feature size
        total_features = main_text_features + section_features + binary_features + numerical_features + medical_concept_features
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer for multi-label classification
        self.classifier = nn.Linear(256, num_labels)
        
        # Learnable thresholds for each label
        self.thresholds = nn.Parameter(torch.ones(num_labels) * 0.2)  # Initialize with 0.2
    
    def forward(self, input_ids, attention_mask, section_texts=None, binary_indicators=None, 
               numerical_features=None, medical_concept_features=None):
        """
        Forward pass of the enhanced model.
        
        Args:
            input_ids: Token IDs from BERT tokenizer for main text
            attention_mask: Attention mask from BERT tokenizer for main text
            section_texts: Dict of section texts with their input_ids and attention_masks
            binary_indicators: Binary indicators for section presence
            numerical_features: Numerical features for section lengths and word counts
            medical_concept_features: Medical concept features
        
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
            
        # Process medical concept features if provided
        if medical_concept_features is not None and self.medical_concept_layer is not None:
            medical_concept_features_processed = self.medical_concept_layer(medical_concept_features)
        else:
            medical_concept_features_processed = torch.zeros(input_ids.size(0), 0, device=input_ids.device)
        
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
        if medical_concept_features_processed.size(1) > 0:
            all_features.append(medical_concept_features_processed)
        
        combined_features = torch.cat(all_features, dim=1)
        
        # Apply feature fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Apply dropout for regularization
        fused_features = self.dropout(fused_features)
        
        # Apply final classification layer
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_thresholds(self):
        """
        Get the learnable thresholds for each label.
        Returns:
            thresholds: Tensor of thresholds for each label
        """
        # Apply sigmoid to keep thresholds in [0, 1] range
        # Cap the range to [0.1, 0.3] as requested
        return torch.sigmoid(self.thresholds) * 0.2 + 0.1  # Range from 0.1 to 0.3

class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted binary cross-entropy loss with logits for multi-label classification.
    """
    def __init__(self, pos_weight=None, reduction='mean'):
        """
        Initialize the weighted loss.
        
        Args:
            pos_weight: Tensor of positive class weights for each label
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Forward pass of the weighted loss.
        
        Args:
            logits: Raw prediction scores from the model
            targets: Ground truth labels
        
        Returns:
            loss: Weighted BCE loss
        """
        # Standard BCEWithLogitsLoss with positive class weights
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )
        
        return loss

def extract_medical_concepts(texts, nlp_model=None):
    """
    Extract medical concepts from texts using ScispaCy or other NLP tools.
    
    Args:
        texts: List of clinical texts
        nlp_model: Loaded NLP model (if None, attempt to load)
    
    Returns:
        dict: Dictionary of extracted medical concepts for each text
    """
    if nlp_model is None:
        try:
            # Try to load a scientific NLP model - use a smaller model to save memory
            nlp_model = spacy.load("en_core_sci_sm")
            logger.info("Loaded scientific NLP model for medical concept extraction")
        except Exception as e:
            logger.warning(f"Failed to load scientific NLP model: {e}")
            logger.info("Using basic approach for medical terms extraction")
            return extract_medical_terms_basic(texts)
    
    medical_concepts = {}
    
    # Common clinical terms and abbreviations
    medical_term_patterns = [
        r'\b(?:history|hx) of\b',
        r'\b(?:diagnosis|dx)\b',
        r'\b(?:treatment|tx)\b',
        r'\b(?:symptoms|sx)\b',
        r'\bhypertension\b|\bhtn\b',
        r'\bdiabetes\b|\bdm\b',
        r'\b(?:coronary artery disease|cad)\b',
        r'\b(?:chronic obstructive pulmonary disease|copd)\b',
        r'\b(?:congestive heart failure|chf)\b',
        r'\b(?:myocardial infarction|mi)\b',
        r'\b(?:cardiovascular accident|cva|stroke)\b'
    ]
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in medical_term_patterns]
    
    # Process each text
    for i, text in tqdm(enumerate(texts), desc="Extracting medical concepts", total=len(texts)):
        if not text or not isinstance(text, str):
            medical_concepts[i] = []
            continue
        
        # Extract entity mentions using NLP model
        doc = nlp_model(text[:10000])  # Limit text size to avoid memory issues
        
        # Extract entities and their types
        extracted_concepts = []
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "CHEMICAL", "PROCEDURE", "ANATOMY"]:
                extracted_concepts.append(ent.text.lower())
        
        # Add pattern-based extractions
        for pattern in patterns:
            matches = pattern.findall(text)
            extracted_concepts.extend([match.lower() for match in matches])
        
        # Get unique concepts
        unique_concepts = list(set(extracted_concepts))
        
        # Store as a sparse vector
        medical_concepts[i] = unique_concepts
    
    # Convert to a feature matrix
    return convert_concepts_to_features(medical_concepts)

def extract_medical_terms_basic(texts, concept_to_idx=None):
    """
    Extract medical terms using a basic regex-based approach.
    
    Args:
        texts: List of clinical texts
    
    Returns:
        dict: Dictionary of extracted medical terms for each text
    """
    medical_terms = {}
    
    # Patterns for common medical terms, conditions, medications
    medical_patterns = [
        r'\b(?:history|hx) of\b',
        r'\b(?:diagnosis|dx)\b',
        r'\b(?:treatment|tx)\b',
        r'\b(?:symptoms|sx)\b',
        r'\bhypertension\b|\bhtn\b',
        r'\bdiabetes\b|\bdm\b',
        r'\b(?:coronary artery disease|cad)\b',
        r'\b(?:chronic obstructive pulmonary disease|copd)\b',
        r'\b(?:congestive heart failure|chf)\b',
        r'\b(?:myocardial infarction|mi)\b',
        r'\b(?:cardiovascular accident|cva|stroke)\b',
        r'\bcancer\b|\bmalignancy\b',
        r'\bpneumonia\b',
        r'\binfection\b|\bsepsis\b',
        r'\brenal failure\b|\bkidney disease\b',
        r'\basthma\b',
        r'\bhyperlipidemia\b',
        r'\bharma\b|\bmedication\b',
        r'\bsurgery\b|\boperation\b',
        r'\banemia\b',
        r'\bthyroid\b',
        r'\bliver\b|\bhepatitis\b',
        r'\bgastritis\b|\bulcer\b',
        r'\bdepression\b|\banxiety\b',
        r'\baddmitted\b|\bdischarged\b'
    ]
    
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in medical_patterns]
    
    # Process each text
    for i, text in tqdm(enumerate(texts), desc="Extracting medical terms", total=len(texts)):
        if not text or not isinstance(text, str):
            medical_terms[i] = []
            continue
            
        # Extract terms based on patterns
        extracted_terms = []
        for pattern in patterns:
            matches = pattern.findall(text)
            extracted_terms.extend([match.lower() for match in matches])
        
        # Get unique terms
        unique_terms = list(set(extracted_terms))
        
        # Store the terms
        medical_terms[i] = unique_terms
    
    # Convert to a feature matrix
    return convert_concepts_to_features(medical_terms, concept_to_idx)

def convert_concepts_to_features(concepts_dict, concept_to_idx=None, max_features=200):
    """
    Convert extracted medical concepts to feature vectors.
    
    Args:
        concepts_dict: Dictionary of extracted concepts for each text
        concept_to_idx: Predefined mapping from concepts to indices (optional)
        max_features: Maximum number of features to include
    
    Returns:
        tuple: (Dictionary of feature vectors, Concept mapping dictionary)
    """
    # If no predefined mapping, create a new one from the data
    if concept_to_idx is None:
        # Collect all unique concepts
        all_concepts = set()
        for concepts in concepts_dict.values():
            all_concepts.update(concepts)
        
        # If too many concepts, select top ones by frequency
        if len(all_concepts) > max_features:
            concept_counts = Counter()
            for concepts in concepts_dict.values():
                concept_counts.update(concepts)
            
            # Keep top N concepts
            top_concepts = [concept for concept, _ in concept_counts.most_common(max_features)]
            all_concepts = set(top_concepts)
        
        # Create a mapping from concept to index
        concept_to_idx = {concept: i for i, concept in enumerate(all_concepts)}
    
    # Create feature vectors
    feature_vectors = {}
    # Use the length of the concept mapping to ensure consistent dimensions
    feature_size = len(concept_to_idx) if concept_to_idx else 1
    
    for text_idx, text_concepts in concepts_dict.items():
        # Create a sparse vector of the correct size
        features = np.zeros(feature_size)
        for concept in text_concepts:
            if concept in concept_to_idx:
                features[concept_to_idx[concept]] = 1.0
        
        feature_vectors[text_idx] = features
    
    # If there are no concepts, use a dummy feature
    if not concept_to_idx:
        for text_idx in concepts_dict.keys():
            feature_vectors[text_idx] = np.zeros(1)
    
    return feature_vectors, concept_to_idx

def generate_pos_weights(labels, scaling_factor=2.0):
    """
    Generate positive class weights for each label based on class imbalance.
    
    Args:
        labels: Array of one-hot encoded labels
        scaling_factor: Factor to scale the weights
    
    Returns:
        tensor: Tensor of positive class weights for each label
    """
    # Calculate the frequency of each label
    label_counts = np.sum(labels, axis=0)
    
    # Calculate the positive class weights
    num_samples = labels.shape[0]
    pos_weights = np.ones_like(label_counts)
    
    # For each label, calculate the weight
    for i in range(len(label_counts)):
        # If the label is not present in any sample, set weight to 1
        if label_counts[i] == 0:
            pos_weights[i] = 1.0
        else:
            # Weight is inverse of the frequency, scaled by a factor
            neg_count = num_samples - label_counts[i]
            pos_weights[i] = (neg_count / label_counts[i]) * scaling_factor
    
    # Convert to tensor
    pos_weights = torch.tensor(pos_weights, dtype=torch.float)
    
    return pos_weights

def apply_smote(X, y, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE for oversampling minority classes in multi-label classification.
    
    Args:
        X: Features
        y: Multi-label targets
        sampling_strategy: Sampling strategy
        random_state: Random seed
    
    Returns:
        tuple: Oversampled features and targets
    """
    logger.info("Applying SMOTE for minority class oversampling")
    
    # Check if any minority class has too few samples for SMOTE
    # SMOTE requires at least 6 samples of the minority class
    label_counts = np.sum(y, axis=0)
    min_samples_required = 6
    
    # If any class has too few samples, use random oversampling instead
    if np.any(label_counts < min_samples_required):
        logger.info(f"Some classes have fewer than {min_samples_required} samples, using RandomOverSampler instead")
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_res, y_res = ros.fit_resample(X, y)
    else:
        try:
            # Apply SMOTE for oversampling
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
            X_res, y_res = smote.fit_resample(X, y)
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using RandomOverSampler as fallback.")
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
            X_res, y_res = ros.fit_resample(X, y)
    
    return X_res, y_res

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
    for key in ['input_ids', 'attention_mask', 'binary_indicators', 'numerical_features', 'medical_concept_features']:
        if key in keys:
            result[key] = torch.stack([sample[key] for sample in batch])
    
    # Collate labels if present
    if 'labels' in keys:
        result['labels'] = torch.stack([sample['labels'] for sample in batch])
    
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

def train_epoch(model, data_loader, optimizer, criterion, device, scheduler=None, 
               gradient_accumulation_steps=1):
    """
    Train the enhanced model for one epoch with gradient accumulation.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on (CPU or GPU)
        scheduler: Learning rate scheduler (optional)
        gradient_accumulation_steps: Number of steps to accumulate gradients
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Track accumulated steps
    steps = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
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
        if 'binary_indicators' in batch and batch['binary_indicators'].numel() > 0:
            binary_indicators = batch['binary_indicators'].to(device)
        
        # Move numerical_features to device if present
        numerical_features = None
        if 'numerical_features' in batch and batch['numerical_features'].numel() > 0:
            numerical_features = batch['numerical_features'].to(device)
            
        # Move medical_concept_features to device if present
        medical_concept_features = None
        if 'medical_concept_features' in batch and batch['medical_concept_features'].numel() > 0:
            medical_concept_features = batch['medical_concept_features'].to(device)
        
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            section_texts=section_texts,
            binary_indicators=binary_indicators,
            numerical_features=numerical_features,
            medical_concept_features=medical_concept_features
        )
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Track loss
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Increment steps
        steps += 1
        
        # Update weights if we've accumulated enough gradients
        if steps % gradient_accumulation_steps == 0:
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"})
    
    # Make sure any remaining gradients are applied
    if steps % gradient_accumulation_steps != 0:
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the enhanced model on validation or test data.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on (CPU or GPU)
    
    Returns:
        tuple: Dictionary of metrics, predictions, and labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_thresholds = []
    
    # Get the model's thresholds
    thresholds = model.get_thresholds().detach().cpu().numpy()
    
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
            if 'binary_indicators' in batch and batch['binary_indicators'].numel() > 0:
                binary_indicators = batch['binary_indicators'].to(device)
            
            # Move numerical_features to device if present
            numerical_features = None
            if 'numerical_features' in batch and batch['numerical_features'].numel() > 0:
                numerical_features = batch['numerical_features'].to(device)
                
            # Move medical_concept_features to device if present
            medical_concept_features = None
            if 'medical_concept_features' in batch and batch['medical_concept_features'].numel() > 0:
                medical_concept_features = batch['medical_concept_features'].to(device)
            
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                section_texts=section_texts,
                binary_indicators=binary_indicators,
                numerical_features=numerical_features,
                medical_concept_features=medical_concept_features
            )
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            # Store predictions and labels
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_thresholds.append(np.tile(thresholds, (preds.shape[0], 1)))
    
    # Concatenate batch results
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_thresholds = np.vstack(all_thresholds)
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    
    # Calculate AUC-ROC (area under the ROC curve)
    auc_scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:  # Only calculate if both classes are present
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
    
    avg_auc = np.mean(auc_scores) if auc_scores else 0
    
    # Convert to binary predictions using learned thresholds
    binary_preds = (all_preds >= all_thresholds).astype(int)
    
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
        'f1': f1,
        'thresholds': thresholds.tolist()
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

def save_model_info(model, section_names, binary_columns, numerical_columns, 
                   medical_concept_size, thresholds, save_path):
    """
    Save model configuration and feature information.
    
    Args:
        model: The neural network model
        section_names: List of section names
        binary_columns: List of binary indicator columns
        numerical_columns: List of numerical feature columns
        medical_concept_size: Size of medical concept features
        thresholds: Learned thresholds for each label
        save_path: Path to save the information
    """
    model_info = {
        'section_names': section_names,
        'binary_columns': binary_columns,
        'numerical_columns': numerical_columns,
        'medical_concept_size': medical_concept_size,
        'thresholds': thresholds,
        'model_parameters': {
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    with open(save_path / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=4)

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

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train an ultra-enhanced BERT-CNN model for ICD-9 code prediction")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory containing the data")
    parser.add_argument("--output_dir", type=str, default="../models/ultra_enhanced_bert_icd9", help="Directory to save the model and results")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for BERT")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for regularization")
    
    # Feature selection arguments
    parser.add_argument("--use_section_texts", action="store_true", help="Whether to use section texts as features")
    parser.add_argument("--use_binary_indicators", action="store_true", help="Whether to use binary indicators as features")
    parser.add_argument("--use_numerical_features", action="store_true", help="Whether to use numerical features as features")
    parser.add_argument("--use_medical_concepts", action="store_true", help="Whether to extract and use medical concepts")
    parser.add_argument("--text_column", type=str, default="summary_snippet_clean", 
                       help="Column to use for main text (clinical_weighted_text or summary_snippet_clean)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                       help="Number of steps to accumulate gradients")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_smote", action="store_true", help="Whether to use SMOTE for minority class oversampling")
    parser.add_argument("--pos_weight_scale", type=float, default=2.0, 
                       help="Scaling factor for positive class weights")
    
    # Thresholding arguments
    parser.add_argument("--init_threshold", type=float, default=0.2, 
                       help="Initial threshold for classification")
    
    return parser.parse_args()

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
    save_dir=None,
    gradient_accumulation_steps=1
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
        gradient_accumulation_steps: Number of steps to accumulate gradients
    
    Returns:
        dict: Training history and best metrics
    """
    # Create timestamp for unique run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ultra_enhanced_bert_cnn_icd9_{timestamp}"
    
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
        'val_f1': [],
        'thresholds': []
    }
    
    # Initialize early stopping variables
    best_val_f1 = 0  # Using F1 as our primary metric
    best_model_state = None
    best_epoch = 0
    best_thresholds = None
    no_improvement_count = 0
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler, 
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Evaluate on validation set
        val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Save the thresholds
        history['thresholds'].append(val_metrics['thresholds'])
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        logger.info(f"Current thresholds: {np.mean(val_metrics['thresholds']):.4f} (mean)")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Check if this is the best model so far (based on F1)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            best_thresholds = val_metrics['thresholds']
            no_improvement_count = 0
            
            # Save best model if save_dir is specified
            if save_dir:
                torch.save(model.state_dict(), save_path / "best_model.pt")
                
                # Save predictions for best model
                np.save(save_path / "best_val_preds.npy", val_preds)
                np.save(save_path / "best_val_labels.npy", val_labels)
                
                # Save best thresholds
                with open(save_path / "best_thresholds.json", 'w') as f:
                    json.dump({'thresholds': best_thresholds}, f, indent=4)
                
                # Save metrics for best model
                with open(save_path / "best_metrics.json", 'w') as f:
                    json.dump({
                        'epoch': best_epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_metrics['loss'],
                        'val_auc': val_metrics['auc'],
                        'val_precision': val_metrics['precision'],
                        'val_recall': val_metrics['recall'],
                        'val_f1': val_metrics['f1'],
                        'thresholds': best_thresholds
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
        'best_val_f1': best_val_f1,
        'best_thresholds': best_thresholds,
        'run_name': run_name
    }

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
    run_dir = output_dir / f"ultra_enhanced_run_{timestamp}"
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
    
    logger.info(f"Found {len(icd9_codes)} ICD-9 codes: {', '.join(icd9_codes[:10])}... (and {len(icd9_codes) - 10} more)")
    
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
    
    # Save the ICD-9 codes as JSON for compatibility with evaluate script
    with open(output_dir / "icd9_codes.json", 'w') as f:
        json.dump(icd9_codes, f, indent=4)
    
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
    
    # Save tokenizer for evaluation
    tokenizer.save_pretrained(output_dir / "tokenizer")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    
    # Prepare datasets
    logger.info("Preparing enhanced datasets...")
    
    # Prepare train dataset
    train_texts, train_section_features, train_binary_indicators, train_numerical_features, train_labels = prepare_enhanced_dataset(
        train_df, icd9_columns, args.text_column, section_names, binary_columns, numerical_columns
    )
    
    # Extract medical concepts if requested
    train_medical_concepts = {}
    concept_to_idx = None
    medical_concept_size = 0

    if args.use_medical_concepts:
        logger.info("Extracting medical concepts from training data...")
        # extract_medical_terms_basic already returns (feature_dict, concept_to_idx)
        train_medical_concepts, concept_to_idx = extract_medical_terms_basic(train_texts)

        # Get medical concept feature size from the training data
        medical_concept_size = len(concept_to_idx) if concept_to_idx else 1
        logger.info(f"Extracted {medical_concept_size} medical concept features")
    
    # Apply SMOTE for minority class oversampling if requested
    if args.use_smote:
        logger.info("Applying SMOTE for minority class oversampling...")
        try:
            # We need to convert the text data to a numeric representation for SMOTE
            # Here we'll just use the presence of medical concepts as features
            X_train = np.zeros((len(train_texts), 1))
            if train_medical_concepts:
                # Use medical concepts as features for oversampling
                X_train = np.array([train_medical_concepts.get(i, np.zeros(1)) for i in range(len(train_texts))])
            
            # Apply SMOTE
            X_train_res, y_train_res = apply_smote(X_train, train_labels)
            
            # We can't directly oversample the text data, so we'll create a mapping
            # from original indices to new indices
            logger.info(f"SMOTE: Original samples: {len(train_texts)}, Oversampled: {len(X_train_res)}")
            
            # Since we can't easily oversample text data, we'll just note that SMOTE was applied
            # In a real implementation, one would need to create synthetic text data or use other methods
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Continuing without oversampling.")
    
    train_dataset = MIMIC_ICD9_Ultra_Enhanced_Dataset(
        texts=train_texts,
        section_features=train_section_features,
        binary_indicators=train_binary_indicators,
        numerical_features=train_numerical_features,
        medical_concepts=train_medical_concepts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Prepare validation dataset
    val_texts, val_section_features, val_binary_indicators, val_numerical_features, val_labels = prepare_enhanced_dataset(
        val_df, icd9_columns, args.text_column, section_names, binary_columns, numerical_columns
    )
    
    # Extract medical concepts from validation data if requested
    val_medical_concepts = {}
    if args.use_medical_concepts:
        logger.info("Extracting medical concepts from validation data...")
        val_medical_concepts, _ = extract_medical_terms_basic(val_texts, concept_to_idx)
    
    val_dataset = MIMIC_ICD9_Ultra_Enhanced_Dataset(
        texts=val_texts,
        section_features=val_section_features,
        binary_indicators=val_binary_indicators,
        numerical_features=val_numerical_features,
        medical_concepts=val_medical_concepts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Prepare test dataset
    test_texts, test_section_features, test_binary_indicators, test_numerical_features, test_labels = prepare_enhanced_dataset(
        test_df, icd9_columns, args.text_column, section_names, binary_columns, numerical_columns
    )
    
    # Extract medical concepts from test data if requested
    test_medical_concepts = {}
    if args.use_medical_concepts:
        logger.info("Extracting medical concepts from test data...")
        test_medical_concepts, _ = extract_medical_terms_basic(test_texts, concept_to_idx)
    
    test_dataset = MIMIC_ICD9_Ultra_Enhanced_Dataset(
        texts=test_texts,
        section_features=test_section_features,
        binary_indicators=test_binary_indicators,
        numerical_features=test_numerical_features,
        medical_concepts=test_medical_concepts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # # Get medical concept feature size
    # medical_concept_size = 0
    # if args.use_medical_concepts:
    #     # Get size from the first item
    #     for idx in train_medical_concepts:
    #         medical_concept_size = len(train_medical_concepts[idx])
    #         break
    
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
    logger.info("Initializing ultra-enhanced model...")
    model = Ultra_Enhanced_BERT_CNN_ICD9(
        num_labels=len(icd9_codes),
        bert_model_name=args.model_name,
        dropout_rate=args.dropout_rate,
        section_names=section_names,
        num_binary_features=len(binary_columns),
        num_numerical_features=len(numerical_columns),
        num_medical_concept_features=medical_concept_size
    )
    model.to(device)
    
    # Calculate positive class weights
    logger.info("Calculating positive class weights...")
    pos_weights = generate_pos_weights(train_labels, scaling_factor=args.pos_weight_scale)
    pos_weights = pos_weights.to(device)
    
    # Initialize weighted loss function
    criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize scheduler with warmup
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
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
        save_dir=run_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
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
    
    # Save best thresholds
    best_thresholds = training_results.get('best_thresholds', [args.init_threshold] * len(icd9_codes))
    with open(output_dir / "best_thresholds.json", 'w') as f:
        json.dump({'thresholds': best_thresholds}, f, indent=4)
    
    # Save model info
    save_model_info(
        model=model,
        section_names=section_names,
        binary_columns=binary_columns,
        numerical_columns=numerical_columns,
        medical_concept_size=medical_concept_size,
        thresholds=best_thresholds,
        save_path=run_dir
    )
    
    # Save test predictions and labels for later analysis
    np.save(run_dir / "test_preds.npy", test_preds)
    np.save(run_dir / "test_labels.npy", test_labels)
    
    # Save best model to output directory
    logger.info(f"Saving best model to {output_dir}")
    torch.save(model.state_dict(), output_dir / "ultra_enhanced_best_model.pt")
    
    logger.info(f"Training and evaluation completed. Results saved to {run_dir}")
    
    return run_dir

if __name__ == "__main__":
    main()