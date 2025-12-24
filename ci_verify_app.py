import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. INITIALIZATION & DUMMY TRAINING DATA
# ==========================================
# Simulating the "Active Learning" dataset described in Stage 5 
# In a real app, this would load from a database.
def get_initial_data():
    data = [
        # FAKE examples
        ("Drinking bleach cures the virus instantly.", "Health", "Fake"),
        ("The earth is flat and NASA is lying.", "Science", "Fake"),
        ("Government hides alien base in Antarctica.", "Politics", "Fake"),
        ("Eating 50 bananas a day guarantees immortality.", "Health", "Fake"),
        
        # REAL examples
        ("Water is essential for human survival.", "Health", "Real"),
        ("The earth revolves around the sun.", "Science", "Real"),
        ("The election is scheduled for November.", "Politics", "Real"),
        ("Regular exercise improves cardiovascular health.", "Health", "Real"),
        
        # Ambiguous/Hard examples (to test confidence thresholds)
        ("New study suggests coffee might have benefits.", "Health", "Real"),
        ("Some officials say the policy is under review.", "Politics", "Real") 
    ]
    return pd.DataFrame(data, columns=["text", "category", "label"])

# Initialize session state for the model and data
if 'df' not in st.session_state:
    st.session_state.df = get_initial_data()

# ==========================================
# 2. MODEL TRAINING (Simulates Stage 3)
# ==========================================
# The document recommends SVM for high-speed triage.
# We use probability=True to get the confidence score needed for Stage 4.

@st.cache_resource(show_spinner=False)
def train_model(data):
    # Stage 2: TF-IDF Vectorization 
    # Stage 3: SVM Classification [cite: 111]
    model = make_pipeline(
        TfidfVectorizer(stop_words='english'),
        SVC(kernel='linear', probability=True, random_state=42)
    )
    model.fit(data['text'], data['label'])
    return model

# Train model on current dataset
model = train_model(st.session_state.df)

# ==========================================
# 3. UTILITY FUNCTIONS (Preprocessing)
# ==========================================
def preprocess_text(text):
    # Simulates Stage 2: Cleaning and Noise Removal [cite: 41]
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

# ==========================================
# 4. USER INTERFACE
# ==========================================
st.set_page_config(page_title="CIVerify Prototype", layout="wide")

st.title("CIVerify: Hybrid Fact-Checking Platform")
st.markdown("""
*Based on the 'Filter-Then-Verify' workflow.*
""")

# --- SIDEBAR: Category Selection (User Interface Requirement 2) ---
st.sidebar.header("Configuration")
category = st.sidebar.selectbox(
    "Select Context/Category",
    ["General", "Health", "Politics", "Science"]
)
st.sidebar.info(f"Current Model: Support Vector Machine (SVM) [cite: 113]")

# --- MAIN AREA: Data Input (User Interface Requirement 1) ---
st.subheader("1. Data Ingestion")
user_input = st.text_area("Enter a claim, news headline, or text snippet:", height=100)

if st.button("Verify Content"):
    if user_input:
        # --- STAGE 2: Pre-processing ---
        clean_text = preprocess_text(user_input)
        
        # --- STAGE 3: Automated Triage & Classification ---
        # Get prediction and probabilities
        prediction = model.predict([clean_text])[0]
        probs = model.predict_proba([clean_text])[0]
        
        # Get max confidence score
        confidence_score = np.max(probs)
        confidence_percent = confidence_score * 100
        
        # --- STAGE 4: Confidence Evaluation & Decision Routing [cite: 115] ---
        # Threshold is set to 90% as per document 
        THRESHOLD = 0.90
        
        st.divider()
        st.subheader("2. Triage Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Detected Category", category)
        
        with col2:
            st.metric("AI Confidence Score", f"{confidence_percent:.2f}%")
            
        with col3:
            # Routing Logic
            if confidence_score >= THRESHOLD:
                status = "AUTO-LABELED"
                color = "green"
            else:
                status = "SENT TO HUMAN REVIEW"
                color = "orange"
            st.markdown(f"**Routing Decision:** :{color}[{status}]")

        # --- Display Result (User Interface Requirement 3) ---
        st.subheader("3. Final Verdict")
        
        if confidence_score >= THRESHOLD:
            # High Confidence Branch [cite: 90]
            if prediction == "Fake":
                st.error(f"❌ **FAKE** (Confidence: {confidence_percent:.1f}%)")
                st.write("System has automatically flagged this content based on high-confidence patterns.")
            else:
                st.success(f"✅ **REAL** (Confidence: {confidence_percent:.1f}%)")
                st.write("System has verified this content as factual.")
        else:
            # Low Confidence Branch [cite: 92] -> Human Collective Review
            st.warning(f"⚠️ **AMBIGUOUS / PENDING VERIFICATION**")
            st.write(f"Confidence ({confidence_percent:.1f}%) is below the 90% threshold.")
            st.write("This content has been routed to the **Human Collective Queue** for expert review.")
            
            # --- STAGE 5: Simulation of Human Feedback (Active Learning)  ---
            st.divider()
            st.markdown("### 🕵️ Human Expert Zone (Simulation)")
            st.write("As an expert, please verify this ambiguous claim to retrain the model:")
            
            c1, c2 = st.columns(2)
            if c1.button("Mark as REAL"):
                new_data = pd.DataFrame([[user_input, category, "Real"]], columns=["text", "category", "label"])
                st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True)
                st.success("Labeled as REAL. The model will learn from this input.")
                st.rerun() # Refresh to retrain
                
            if c2.button("Mark as FAKE"):
                new_data = pd.DataFrame([[user_input, category, "Fake"]], columns=["text", "category", "label"])
                st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True)
                st.error("Labeled as FAKE. The model will learn from this input.")
                st.rerun() # Refresh to retrain

    else:
        st.warning("Please enter text to verify.")

# --- Footer: System Stats ---
st.divider()
st.caption(f"Knowledge Base Size: {len(st.session_state.df)} verified claims | Active Learning Enabled")