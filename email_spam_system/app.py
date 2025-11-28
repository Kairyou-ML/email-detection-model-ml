import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'models/distilbert-spam'

# ==========================================
# 1. LOAD MODEL & TOKENIZER
# ==========================================
@st.cache_resource
def load_model():
    """
    Loads the model and tokenizer only once to improve app speed.
    """
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def predict_spam(text):
    """
    Returns probability of Spam (0.0 to 1.0) and the predicted label.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities (Softmax)
    probs = F.softmax(outputs.logits, dim=1)
    spam_prob = probs[0][1].item() # Index 1 is Spam
    
    return spam_prob

def explain_prediction(text, original_prob):
    """
    EXPLAINABILITY (Bonus):
    Identifies the top 3 words contributing to the Spam decision.
    Method: 'Masking' - We remove one word at a time and see how much the spam score drops.
    """
    words = text.split()
    # Limit analysis to first 50 words for speed
    words = words[:50] 
    word_scores = []
    
    for i in range(len(words)):
        # Create a version of text without this word
        masked_text = " ".join(words[:i] + words[i+1:])
        masked_prob = predict_spam(masked_text)
        
        # Contribution = How much prob drops when word is removed
        contribution = original_prob - masked_prob
        word_scores.append((words[i], contribution))
    
    # Sort by contribution (highest drop first)
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:3]

# ==========================================
# 3. UI LAYOUT
# ==========================================
st.set_page_config(page_title="Spam Detector", page_icon="🛡️")

st.title("🛡️ Intelligent Email Spam Detector")
st.markdown("Enter an email below to check if it's **Safe** or **Spam**.")

col1, col2 = st.columns([3, 1])

with col1:
    subject = st.text_input("Email Subject", placeholder="e.g., You won a lottery!")
    content = st.text_area("Email Content", placeholder="Paste the email body here...", height=150)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2058/2058138.png", width=100)
    st.write("### AI Engine")
    st.caption(f"Model: DistilBERT")

# ==========================================
# 4. ANALYSIS LOGIC
# ==========================================
if st.button("Analyze Email", type="primary"):
    if not content:
        st.warning("Please enter some email content to analyze.")
    else:
        # Combine text (Simple concatenation matches training logic)
        full_text = f"{subject} {content}".strip()
        
        with st.spinner("🔍 Scanning email patterns..."):
            spam_probability = predict_spam(full_text)
        
        # Display Result
        st.divider()
        col_res, col_score = st.columns([2, 1])
        
        is_spam = spam_probability > 0.5
        
        with col_res:
            if is_spam:
                st.error("🚨 **SPAM DETECTED**")
                st.write("This email contains patterns commonly found in spam.")
            else:
                st.success("✅ **LOOKS SAFE (HAM)**")
                st.write("This email appears legitimate.")
        
        with col_score:
            st.metric("Confidence Score", f"{spam_probability:.1%}", 
                      delta="High Risk" if is_spam else "Low Risk",
                      delta_color="inverse" if is_spam else "normal")

        # Explainability Section
        if is_spam:
            st.subheader("🧐 Why is this Spam?")
            with st.spinner("Identifying trigger words..."):
                top_contributors = explain_prediction(full_text, spam_probability)
            
            st.write("These words contributed most to the decision:")
            cols = st.columns(3)
            for idx, (word, score) in enumerate(top_contributors):
                if score > 0.01: # Only show significant contributors
                    cols[idx].error(f"{word}")
            
            st.caption(f"*Based on perturbation analysis of the top {len(full_text.split())} words.*")

# Footer
st.markdown("---")
st.caption("Capstone Project | Powered by DistilBERT & Streamlit")