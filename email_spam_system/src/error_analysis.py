import pandas as pd
import torch
import os
import matplotlib
# Force non-interactive backend to avoid Tcl/Tk errors on Windows/Servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your BEST model (DistilBERT)
MODEL_PATH = 'models/distilbert-spam' 
TEST_DATA_PATH = 'data/test_processed.csv'
REPORT_DIR = 'reports/'

class SimpleDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])

def analyze_errors():
    print("🕵️ Starting Deep Error Analysis on Best Model (DistilBERT)...")
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 1. Load Model & Tokenizer
    # ------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}.")
        print("   Make sure you unzipped the best model from the training step.")
        return

    try:
        print(f"   Loading model & tokenizer from {MODEL_PATH}...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Load Test Data
    # ------------------------------------------------
    print("   Loading test data...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Error: Test data not found at {TEST_DATA_PATH}.")
        return

    df = pd.read_csv(TEST_DATA_PATH)
    
    # Ensure text is string and handle missing values
    text_col = 'text' if 'text' in df.columns else 'clean_text'
    texts = df[text_col].fillna("").astype(str).tolist()
    
    # Encode labels (Ham=0, Spam=1)
    le = LabelEncoder()
    true_labels = le.fit_transform(df['label']) 
    class_names = le.classes_ # Should be ['ham', 'spam']

    # 3. Generate Predictions
    # ------------------------------------------------
    print("   Running predictions (this might take a moment)...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    dataset = SimpleDataset(encodings)
    
    trainer = Trainer(model=model)
    preds_output = trainer.predict(dataset)
    
    # Get predicted classes and confidence scores
    pred_labels = preds_output.predictions.argmax(-1)
    probs = torch.nn.functional.softmax(torch.tensor(preds_output.predictions), dim=-1)
    confidence = probs.max(dim=1).values.numpy()

    # Add columns to dataframe for analysis
    df['true_label'] = true_labels
    df['pred_label'] = pred_labels
    df['confidence'] = confidence

    # 4. Confusion Matrix Heatmap
    # ------------------------------------------------
    print("📊 Generating Confusion Matrix...")
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: DistilBERT')
    
    cm_path = os.path.join(REPORT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"   ✅ Saved heatmap to: {cm_path}")

    # 5. Extract False Positives (Safe -> Marked as Spam)
    # ------------------------------------------------
    # High Precision is key, so we need to analyze these deeply
    fps = df[(df['true_label'] == 0) & (df['pred_label'] == 1)].sort_values(by='confidence', ascending=False)
    
    print(f"\n🚨 FALSE POSITIVES (Safe emails classified as Spam): {len(fps)}")
    print("   Why this happens: Aggressive keywords ('free', 'money'), weird formatting, or sarcasm.")
    print("-" * 60)
    for i, row in fps.head(5).iterrows():
        print(f"🔹 Confidence: {row['confidence']:.2f}")
        print(f"   Text: \"{row[text_col]}\"")
        print("-" * 60)

    # 6. Extract False Negatives (Spam -> Marked as Ham)
    # ------------------------------------------------
    # These are spam emails that sneaked through
    fns = df[(df['true_label'] == 1) & (df['pred_label'] == 0)].sort_values(by='confidence', ascending=False)
    
    print(f"\n🕵️ FALSE NEGATIVES (Spam emails classified as Safe): {len(fns)}")
    print("   Why this happens: Short text, lack of keywords, or 'conversational' spam.")
    print("-" * 60)
    for i, row in fns.head(5).iterrows():
        print(f"🔸 Confidence: {row['confidence']:.2f}")
        print(f"   Text: \"{row[text_col]}\"")
        print("-" * 60)

    # 7. Model Status Check
    # ------------------------------------------------
    print("\n💾 Model & Vectorizer Status")
    # For Transformers, the 'Vectorizer' is the 'Tokenizer'
    if os.path.exists(os.path.join(MODEL_PATH, 'config.json')) and \
       os.path.exists(os.path.join(MODEL_PATH, 'vocab.txt')):
        print("   ✅ Best Model (DistilBERT) and Tokenizer are already saved in:")
        print(f"      {MODEL_PATH}")
        print("   ready for the Application Phase.")
    else:
        print("   ⚠️ Warning: Model files seem incomplete. Please re-run training.")

if __name__ == "__main__":
    analyze_errors()