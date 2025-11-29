import pandas as pd
import string
import re
import spacy
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# Attempt to load spaCy; gracefully handle if not present
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    # Fallback to a blank model to prevent immediate crash, though lemmatization won't work well
    nlp = spacy.blank("en")

class EmailPipeline:
    """
    A comprehensive pipeline for processing email data.
    Encapsulates Feature Engineering, Text Cleaning, and Data Splitting.
    """

    def __init__(self):
        # We load stopwords once to speed up processing
        self.stopwords = nlp.Defaults.stop_words

    def _count_cap_words(self, text):
        """Helper: Counts words that are fully capitalized (e.g., 'FREE', 'URGENT')."""
        if not isinstance(text, str): return 0
        return sum(1 for word in text.split() if word.isupper() and len(word) > 1)

    def add_manual_features(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Step 1: Feature Engineering.
        Extracts structural features BEFORE cleaning the text.
        """
        print("Extracting manual features...")
        df = df.copy()
        
        # 1. Body Length: Length of the raw text
        df['body_len'] = df[text_col].apply(lambda x: len(str(x)))
        
        # 2. Punctuation %: (count of punct chars / total len) * 100
        def count_punct(text):
            if not isinstance(text, str) or len(text) == 0: return 0
            count = sum(1 for char in text if char in string.punctuation)
            return round((count / (len(text) - text.count(" "))) * 100, 2)
        
        df['punct%'] = df[text_col].apply(count_punct)
        
        # 3. Capitalized Word Count: Signals shouting/urgency
        df['cap_count'] = df[text_col].apply(self._count_cap_words)
        
        # 4. Has Link: Binary flag for 'http' or 'https'
        df['has_link'] = df[text_col].apply(lambda x: 1 if 'http' in str(x).lower() else 0)
        
        return df

    def clean_text(self, text: str) -> str:
        """
        Step 2: Text Cleaning.
        HTML removal -> Lowercase -> Punctuation removal -> Lemmatization.
        """
        if not isinstance(text, str): return ""

        # A. Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # B. Remove URLs (since we already captured 'has_link')
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # C. Remove Punctuation
        text = "".join([char for char in text if char not in string.punctuation])

        # D. Tokenization & Lemmatization (using spaCy)
        doc = nlp(text.lower())
        
        # Keep tokens that are NOT stopwords and NOT whitespace
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
        
        return " ".join(tokens)

    def preprocess(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Orchestrates the full preprocessing workflow.
        """
        # 1. Feature Engineering
        df_featured = self.add_manual_features(df, text_col)
        
        # 2. Text Cleaning
        print("🧹 Cleaning text and lemmatizing (this may take a moment)...")
        df_featured['clean_text'] = df_featured[text_col].apply(self.clean_text)
        
        return df_featured

    def split_data(self, df: pd.DataFrame, target_col: str, test_size=0.2):
        """
        Step 3: Stratified Split.
        Ensures the ratio of Spam/Ham is preserved in both Train and Test sets.
        """
        print("Splitting data (80/20 Stratified)...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)