import pandas as pd
import numpy as np
import joblib
import os

# Scikit-learn & Imbalanced-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# SMOTE for imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Local imports
from src.visualization import plot_roc_curves

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_DATA_PATH = 'data/train_processed.csv'
TEST_DATA_PATH = 'data/test_processed.csv'
MODEL_SAVE_PATH = 'models/'

def load_data(path):
    """Loads CSV and ensures text columns are strings."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    # Fill NaN values in 'clean_text' with empty string
    df['clean_text'] = df['clean_text'].fillna("")
    return df

def train_benchmarking():
    print("🚀 Loading processed data for benchmarking...")
    try:
        train_df = load_data(TRAIN_DATA_PATH)
        test_df = load_data(TEST_DATA_PATH)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Prepare X and y
    features = ['clean_text', 'body_len', 'punct%', 'cap_count']
    
    X_train = train_df[features]
    y_train_raw = train_df['label']
    
    X_test = test_df[features]
    y_test_raw = test_df['label']

    # Encode Labels (Spam = 1, Ham = 0)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    print(f"   Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 1. Define Preprocessing Pipeline
    # ----------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=3000), 'clean_text'), # Reduced features slightly for speed
            ('num', 'passthrough', ['body_len', 'punct%', 'cap_count'])
        ]
    )

    # 2. Define Models
    # ----------------------------------------------------
    models = {
        "Naive Bayes": ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', MultinomialNB())
        ]),
        
        # CHANGED: Switched to LinearSVC for drastic speed improvement
        # Note: This model will likely be skipped in ROC plotting because it lacks predict_proba
        "SVM (Linear)": ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LinearSVC(random_state=42, dual='auto', max_iter=1000)) 
        ]),
        
        "Random Forest": ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)) # Reduced trees for speed
        ])
    }

    # 3. Training Loop & Evaluation
    # ----------------------------------------------------
    results_list = []
    trained_models = {}

    for name, pipeline in models.items():
        print(f"\n🧠 Training {name}...")
        
        # Fit model
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    # 4. Comparison Table
    # ----------------------------------------------------
    results_df = pd.DataFrame(results_list).set_index("Model")
    print("\n🏆 Model Benchmarking Results:")
    print("-" * 60)
    print(results_df.round(4))
    print("-" * 60)

    # 5. Recommendation for Precision
    # ----------------------------------------------------
    best_precision_model = results_df['Precision'].idxmax()
    best_precision_val = results_df['Precision'].max()
    
    print(f"\n💡 Best Model for Precision (Avoiding False Positives): {best_precision_model}")
    print(f"   Precision Score: {best_precision_val:.4f}")
    
    # Save best model logic
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    joblib.dump(trained_models[best_precision_model], os.path.join(MODEL_SAVE_PATH, 'best_model.pkl'))
    print(f"   Saved {best_precision_model} to {MODEL_SAVE_PATH}")

    # 6. Plot ROC Curve
    # ----------------------------------------------------
    print("\n📊 Generating ROC Curves...")
    # This will now safely skip LinearSVC (which has no probabilities) and plot the others
    plot_roc_curves(trained_models, X_test, y_test)

if __name__ == "__main__":
    train_benchmarking()