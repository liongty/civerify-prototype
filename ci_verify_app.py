import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. INITIAL DATA (SIMULATED KNOWLEDGE BASE)
# ==========================================
def get_initial_data():
    data = [
        # FAKE
        ("Drinking bleach cures the virus instantly.", "Health", "Fake"),
        ("The earth is flat and NASA is lying.", "Science", "Fake"),
        ("Government hides alien base in Antarctica.", "Politics", "Fake"),
        ("Eating 50 bananas a day guarantees immortality.", "Health", "Fake"),

        # REAL
        ("Water is essential for human survival.", "Health", "Real"),
        ("The earth revolves around the sun.", "Science", "Real"),
        ("The election is scheduled for November.", "Politics", "Real"),
        ("Regular exercise improves cardiovascular health.", "Health", "Real"),

        # AMBIGUOUS
        ("New study suggests coffee might have benefits.", "Health", "Real"),
        ("Some officials say the policy is under review.", "Politics", "Real")
    ]

    return pd.DataFrame(data, columns=["text", "category", "label"])

if "df" not in st.session_state:
    st.session_state.df = get_initial_data()

# ==========================================
# 2. TEXT PREPROCESSING (USED EVERYWHERE)
# ==========================================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def prepare_features(df: pd.DataFrame) -> pd.Series:
    """
    Combine category + text and apply preprocessing
    """
    combined = df["category"] + " " + df["text"]
    return combined.apply(preprocess_text)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X = prepare_features(df)
    y = df["label"]

    model = make_pipeline(
        TfidfVectorizer(stop_words="english"),
        SVC(kernel="linear", probability=True, random_state=42)
    )

    model.fit(X, y)
    return model

# Train initial model
model = train_model(st.session_state.df)

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="CIVerify Prototype", layout="wide")

st.title("CIVerify: Hybrid Fact-Checking Platform")
st.markdown("*Based on the Filter-Then-Verify workflow*")

# ---------- Sidebar ----------
st.sidebar.header("Configuration")
category = st.sidebar.selectbox(
    "Select Context / Category",
    ["Health", "Politics", "Science", "General"]
)

st.sidebar.info("Model: TF-IDF + Linear SVM")

# ---------- Input ----------
st.subheader("1. Data Ingestion")
user_input = st.text_area(
    "Enter a claim, news headline, or text snippet:",
    height=120
)

# ==========================================
# 5. VERIFICATION PIPELINE
# ==========================================
if st.button("Verify Content"):
    if not user_input.strip():
        st.warning("Please enter text to verify.")
        st.stop()

    # ----- Stage 2: Preprocessing -----
    # Match the training logic: Category + Text
    combined_input = f"{category} {user_input}"
    clean_input = preprocess_text(combined_input)

    # ----- Stage 3: Prediction -----
    prediction = model.predict([clean_input])[0]
    probabilities = model.predict_proba([clean_input])[0]

    confidence_score = np.max(probabilities)
    confidence_percent = confidence_score * 100

    # ----- Stage 4: Decision Routing -----
    THRESHOLD = 0.90

    st.divider()
    st.subheader("2. Triage Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Detected Category", category)

    with col2:
        st.metric("AI Confidence", f"{confidence_percent:.2f}%")

    with col3:
        if confidence_score >= THRESHOLD:
            st.markdown("**Routing:** :green[AUTO-LABELED]")
        else:
            st.markdown("**Routing:** :orange[SENT TO HUMAN REVIEW]")

    # ----- Stage 5: Final Verdict -----
    st.subheader("3. Final Verdict")

    if confidence_score >= THRESHOLD:
        if prediction == "Fake":
            st.error(f"❌ **FAKE** ({confidence_percent:.1f}%)")
            st.write("Automatically flagged based on high-confidence patterns.")
        else:
            st.success(f"✅ **REAL** ({confidence_percent:.1f}%)")
            st.write("Content verified as factual by the system.")
    else:
        st.warning("⚠️ **AMBIGUOUS / PENDING VERIFICATION**")
        st.write(f"Confidence ({confidence_percent:.1f}%) is below the 90% threshold.")
        st.write("Routed to the **Human Collective Review Queue**.")

        # ----- Active Learning Simulation -----
        st.divider()
        st.markdown("### 🕵️ Human Expert Zone (Simulation)")
        st.write("Please label the claim to improve the system:")

        c1, c2 = st.columns(2)

        if c1.button("Mark as REAL"):
            new_row = pd.DataFrame(
                [[user_input, category, "Real"]],
                columns=["text", "category", "label"]
            )
            st.session_state.df = pd.concat(
                [st.session_state.df, new_row],
                ignore_index=True
            )
            st.success("Labeled as REAL. Model retraining...")
            st.rerun()

        if c2.button("Mark as FAKE"):
            new_row = pd.DataFrame(
                [[user_input, category, "Fake"]],
                columns=["text", "category", "label"]
            )
            st.session_state.df = pd.concat(
                [st.session_state.df, new_row],
                ignore_index=True
            )
            st.error("Labeled as FAKE. Model retraining...")
            st.rerun()

# ---------- Footer ----------
st.divider()
st.caption(
    f"Knowledge Base Size: {len(st.session_state.df)} verified claims | "
    "Active Learning Enabled | Confidence scores are probabilistic estimates"
)
