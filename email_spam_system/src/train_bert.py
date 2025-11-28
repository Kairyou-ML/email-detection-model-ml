import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch import nn

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_PATH = 'data/train_processed.csv'
TEST_PATH = 'data/test_processed.csv'
MODEL_DIR = 'models/distilbert-spam'

# Hyperparameters
MAX_LEN = 128     # Sufficient for most emails
BATCH_SIZE = 8    # Small batch size for Colab/Local GPU
EPOCHS = 2        # Low epochs to prevent overfitting on small data
LEARNING_RATE = 2e-5 # Low LR for fine-tuning pre-trained models

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

class SpamDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for loading emails."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    """Callback to calculate Precision/Recall during training."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

class CustomTrainer(Trainer):
    """
    Custom Trainer to handle Class Imbalance (Spam < Ham).
    We inject a weighted Loss Function directly into the training loop.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Calculate Class Weights: Higher weight for Spam (Index 1)
        # Weight 6.0 roughly balances a 13% spam / 87% ham ratio
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0]).to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def train_distilbert():
    # 1. Load Data
    # ------------------------------------------------
    print("🚀 Loading data for DistilBERT...")
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ Error: {TRAIN_PATH} not found. Run preprocessing first.")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # DistilBERT prefers RAW text (with stopwords/grammar), not the cleaned version.
    # We prioritize 'text' (raw) over 'clean_text'.
    text_col = 'text' if 'text' in train_df.columns else 'clean_text'
    print(f"   Using column: '{text_col}' for training.")
    
    # Handle NaNs just in case
    train_texts = train_df[text_col].fillna("").astype(str).tolist()
    train_labels = train_df['label'].tolist()
    
    test_texts = test_df[text_col].fillna("").astype(str).tolist()
    test_labels = test_df['label'].tolist()

    # Split Train into Train/Validation (90/10)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    # 2. Tokenization
    # ------------------------------------------------
    print("Tokenizing data...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LEN)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LEN)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LEN)

    # Create Datasets
    train_dataset = SpamDataset(train_encodings, train_labels)
    val_dataset = SpamDataset(val_encodings, val_labels)
    test_dataset = SpamDataset(test_encodings, test_labels)

    # 3. Initialize Model
    # ------------------------------------------------
    print("Initializing DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

    # 4. Training Arguments
    # ------------------------------------------------
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        num_train_epochs=EPOCHS,         # Total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # Batch size
        per_device_eval_batch_size=BATCH_SIZE*2,
        learning_rate=LEARNING_RATE,     # 2e-5 as requested
        warmup_steps=100,                # Number of warmup steps
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=50,
        eval_strategy="epoch",           # Evaluate every epoch
        save_strategy="epoch",           # Save checkpoint every epoch
        load_best_model_at_end=True,     # Load the best model when finished
        report_to="none"                 # Disable wandb/mlflow logging
    )

    # 5. Train
    # ------------------------------------------------
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n Starting Training ({EPOCHS} Epochs)...")
    trainer.train()

    # 6. Final Evaluation
    # ------------------------------------------------
    print("\n Evaluating on Test Set...")
    results = trainer.evaluate(test_dataset)
    
    print("\n DistilBERT Results:")
    print("-" * 30)
    print(f"Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall:    {results['eval_recall']:.4f}")
    print(f"F1-Score:  {results['eval_f1']:.4f}")
    print("-" * 30)

    # 7. Save Model
    # ------------------------------------------------
    print(f"Saving model to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

if __name__ == "__main__":
    train_distilbert()