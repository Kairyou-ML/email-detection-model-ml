import pandas as pd
from src.features import EmailPipeline
from src.visualization import plot_feature_distributions

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = 'data/emails.csv'  # Update this path to your actual file
TEXT_COLUMN = 'text'           # The script will try to rename 'v2' to this
LABEL_COLUMN = 'label'         # The script will try to rename 'v1' to this

def main():
    # 1. Load Data (Robust Loading)
    # --------------------------------------
    df = None
    # List of encodings to try. Spam datasets often use 'latin-1' or 'windows-1252'
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            print(f"Trying to read with encoding: {encoding}...")
            df = pd.read_csv(DATA_PATH, encoding=encoding)
            print(f"✅ Data loaded successfully using '{encoding}' encoding. Shape: {df.shape}")
            break
        except UnicodeDecodeError:
            print(f"⚠️ Failed with encoding: {encoding}. Retrying...")
        except FileNotFoundError:
            print(f"❌ Error: File not found at {DATA_PATH}. Please check the path.")
            return

    if df is None:
        print("❌ Critical Error: Could not read the file with any standard encoding.")
        return

    # 1.1 Clean Column Names (Handle v1/v2 common in spam datasets)
    # -------------------------------------------------------------
    print(f"   Raw Columns: {df.columns.tolist()}")
    
    # Standardize 'v1' -> 'label', 'v2' -> 'text' (Common in SMS Spam Collection)
    if 'v1' in df.columns and 'v2' in df.columns:
        print("   Found 'v1' and 'v2' columns. Renaming to 'label' and 'text'.")
        df = df.rename(columns={'v1': LABEL_COLUMN, 'v2': TEXT_COLUMN})
        
        # Drop likely empty columns (Unnamed: 2, 3, 4) often found in this dataset
        drop_cols = [c for c in df.columns if 'Unnamed' in c]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            print(f"   Dropped columns: {drop_cols}")

    # Verify columns exist
    if LABEL_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
        print(f"❌ Error: Expected columns '{LABEL_COLUMN}' and '{TEXT_COLUMN}' not found.")
        print(f"   Available columns: {df.columns.tolist()}")
        print("   Please update TEXT_COLUMN and LABEL_COLUMN in the script configuration.")
        return

    # Quick check to map labels if they are strings
    if df[LABEL_COLUMN].dtype == 'object':
        print(f"   Labels found: {df[LABEL_COLUMN].unique()}")

    # 2. Pipeline Execution
    # --------------------------------------
    pipeline = EmailPipeline()
    
    # Apply Feature Engineering & Cleaning
    try:
        df_processed = pipeline.preprocess(df, text_col=TEXT_COLUMN)
        
        print("\nSample of processed data:")
        print(df_processed[['label', 'body_len', 'punct%', 'cap_count', 'clean_text']].head())

        # 3. Visualization
        # --------------------------------------
        print("\n📊 Generating visualizations...")
        plot_feature_distributions(df_processed, target_col=LABEL_COLUMN)

        # 4. Stratified Split
        # --------------------------------------
        X_train, X_test, y_train, y_test = pipeline.split_data(
            df_processed, 
            target_col=LABEL_COLUMN
        )
        
        print(f"\n✅ Split Complete.")
        print(f"   Training Set: {X_train.shape[0]} samples")
        print(f"   Test Set:     {X_test.shape[0]} samples")
        
        # 5. Save for next step
        # --------------------------------------
        # Ensure 'data' directory exists, though usually handled by user setup
        X_train.join(y_train).to_csv('data/train_processed.csv', index=False)
        X_test.join(y_test).to_csv('data/test_processed.csv', index=False)
        print("💾 Processed datasets saved to 'data/' folder.")

    except Exception as e:
        print(f"\n❌ An error occurred during the pipeline execution:\n{e}")

if __name__ == "__main__":
    main()