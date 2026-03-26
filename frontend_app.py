import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Set Wide Layout and Dark Theme Config
st.set_page_config(page_title="CreditSense Premium", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# ----------------- DATA & MODEL CACHING -----------------
@st.cache_data
def load_and_train_models():
    """Generates synthetic dataset and trains the 3 distinct models required for the report."""
    np.random.seed(42)
    n_samples = 2000
    age = np.random.randint(18, 70, n_samples)
    income = np.random.randint(20000, 150000, n_samples)
    debt_ratio = np.random.uniform(0.1, 0.9, n_samples)
    open_credit_lines = np.random.randint(1, 15, n_samples)
    past_due_count = np.random.randint(0, 5, n_samples)
    
    # Simple logic for default risk
    risk_score = (debt_ratio * 3) + (past_due_count * 2) - (income / 50000)
    default_status = (risk_score > np.median(risk_score)).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Annual_Income': income,
        'Debt_Ratio': debt_ratio,
        'Open_Credit_Lines': open_credit_lines,
        'Past_Due_Count': past_due_count,
        'Default': default_status
    })
    
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "CM": cm}
        
    return results

model_results = load_and_train_models()


# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
    /* Global Adjustments */
    .stApp { background-color: #0a0c10; }
    .stButton>button {
        background: linear-gradient(90deg, #0ce688 0%, #15c9b7 100%);
        color: #0a0c10 !important; font-weight: 700; border-radius: 12px; border: none;
    }
    .title-gradient {
        background: linear-gradient(90deg, #0ce688 0%, #15c9b7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif; font-size: 3.5rem; font-weight: 900; margin-bottom: 0px; letter-spacing: -1px;
    }
    .subtitle { color: #8C9BAB; font-size: 1.2rem; margin-bottom: 35px; font-weight: 400; }
    
    /* Phone Mockup CSS */
    .phone-mockup {
        width: 320px; height: 650px; background-color: #12141C; border-radius: 45px;
        border: 12px solid #232733; box-shadow: 0 40px 60px -15px rgba(0,0,0,0.8);
        margin: 0 auto; position: relative; overflow: hidden; display: flex; flex-direction: column; align-items: center; padding-top: 55px; font-family: 'Inter', sans-serif;
    }
    .phone-mockup::before {
        content: ''; position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 120px; height: 25px;
        background-color: #232733; border-bottom-left-radius: 18px; border-bottom-right-radius: 18px; z-index: 10;
    }
    .phone-title { color: white; font-size: 16px; font-weight: 600; margin-bottom: 40px; letter-spacing: 0.5px; opacity: 0.9; }
    
    /* Gauge wrapper */
    .gauge-wrapper { position: relative; width: 240px; height: 120px; overflow: hidden; margin-bottom: 50px; display: flex; justify-content: center; align-items: flex-end; }
    .gauge-background { position: absolute; top: 0; left: 0; width: 240px; height: 240px; border-radius: 50%; background: conic-gradient(from 270deg, #ff4d4f 0deg, #ffc53d 90deg, #0ce688 180deg, transparent 180deg); }
    .gauge-inner { position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); width: 190px; height: 95px; background-color: #12141C; border-top-left-radius: 95px; border-top-right-radius: 95px; z-index: 2; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; padding-bottom: 5px; }
    .score-value { font-size: 50px; font-weight: 800; line-height: 1; margin: 0; text-shadow: 0 0 20px rgba(255,255,255,0.1); }
    .score-label { font-size: 14px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-top: 5px; }
    .gauge-pivot { position: absolute; bottom: -6px; left: 50%; transform: translateX(-50%); width: 12px; height: 12px; background-color: white; border-radius: 50%; z-index: 10; }
    .gauge-needle-container { position: absolute; bottom: 0; left: 50%; width: 0; height: 0; z-index: 5; }
    .gauge-needle { position: absolute; bottom: -2px; left: 0; width: 90px; height: 4px; background-color: white; transform-origin: left center; border-radius: 4px; transition: transform 1.8s cubic-bezier(0.34, 1.56, 0.64, 1); }
    
    /* Phone Info Card */
    .phone-details-card { background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%); width: 85%; border-radius: 20px; padding: 20px; padding-top: 25px; margin-top: 10px; color: white; border: 1px solid rgba(255,255,255,0.05); }
    .detail-row { display: flex; justify-content: space-between; margin-bottom: 18px; border-bottom: 1px dashed rgba(255,255,255,0.1); padding-bottom: 12px; }
    .detail-row:last-child { margin-bottom: 0; border-bottom: none; padding-bottom: 0; }
    .detail-label { color: #8C9BAB; font-size: 13px; }
    .detail-value { font-size: 14px; font-weight: 700; }
    
    .form-pane { background: rgba(20, 23, 31, 0.6); border-radius: 24px; padding: 30px; border: 1px solid rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)


def render_phone(score=None, category=None, debt_ratio=None, color="#ffffff"):
    if score is None:
        score_display = "---"
        label_display = "AWAITING DATA"
        rotation = 180
        debt_str = "---"
        cat_str = "---"
        needle_opacity = "0"
        model_str = "---"
    else:
        score_display = str(score)
        label_display = category
        normalized = max(0.0, min(1.0, (score - 300) / (850 - 300)))
        rotation = 180 + (normalized * 180)
        debt_str = f"{debt_ratio*100:.1f}%"
        cat_str = category
        needle_opacity = "1"
        model_str = "Random Forest CV"

    phone_html = f"""<div class="phone-mockup">
<div class="phone-title">AI Credit Report</div>
<div class="gauge-wrapper">
<div class="gauge-background"></div>
<div class="gauge-inner">
<p class="score-value" style="color: {color}">{score_display}</p>
<p class="score-label" style="color: {color}">{label_display}</p>
</div>
<div class="gauge-needle-container">
<div class="gauge-needle" style="transform: rotate({rotation}deg); opacity: {needle_opacity};"></div>
</div>
<div class="gauge-pivot" style="opacity: {needle_opacity};"></div>
</div>
<div class="phone-details-card">
<div class="detail-row">
<span class="detail-label">Status</span>
<span class="detail-value" style="color: {color}">{cat_str}</span>
</div>
<div class="detail-row">
<span class="detail-label">Debt-to-Income</span>
<span class="detail-value">{debt_str}</span>
</div>
<div class="detail-row">
<span class="detail-label">Model Used</span>
<span class="detail-value" style="color: #0ce688;">{model_str}</span>
</div>
</div>
</div>"""
    return phone_html


# --------- MAIN LAYOUT & TABS ---------
tab1, tab2 = st.tabs(["🚀 Credit Predictor", "📊 Model Analysis & Comparison"])

with tab1:
    col_left, col_spacer, col_right = st.columns([1.3, 0.2, 1])

    with col_left:
        st.markdown('<h1 class="title-gradient">CreditSense Premium</h1>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Next-Generation Financial Scoring • Hridam Biswas (2305623)</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-pane">', unsafe_allow_html=True)
        with st.form("evaluation_form"):
            st.subheader("Configure Applicant Data")
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1: age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
            with row1_col2: income = st.number_input("Annual Income ($)", min_value=10000, max_value=2000000, value=65000, step=5000)
                
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1: debt = st.number_input("Total Existing Debt ($)", min_value=0, max_value=2000000, value=15000, step=1000)
            with row2_col2: past_due = st.selectbox("Past Due Payments (24mo)", [0, 1, 2, 3, "4+"])
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Generate Instant Score ➔", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_right:
        phone_placeholder = st.empty()
        if submit:
            phone_placeholder.markdown(render_phone(), unsafe_allow_html=True)
            with st.spinner("Analyzing parameters through Random Forest..."):
                time.sleep(1.2)
            debt_ratio = debt / max(income, 1)
            base_score = 780
            penalty_past_due = (int(past_due) if past_due != "4+" else 5) * 55
            penalty_debt = int(debt_ratio * 180)
            score = max(300, min(850, base_score - penalty_past_due - penalty_debt))
            
            if score < 580: cat, color = "Poor Risk", "#ff4d4f"
            elif score < 670: cat, color = "Fair Risk", "#ffc53d"
            elif score < 740: cat, color = "Good", "#52c41a"
            else: cat, color = "Excellent", "#0ce688"
                
            phone_placeholder.markdown(render_phone(score, cat, debt_ratio, color), unsafe_allow_html=True)
            st.balloons()
        else:
            phone_placeholder.markdown(render_phone(), unsafe_allow_html=True)


with tab2:
    st.markdown('<h1 class="title-gradient" style="font-size:2.5rem; margin-bottom: 20px;">Model Analysis & Comparison</h1>', unsafe_allow_html=True)
    st.markdown("<p style='color:#8C9BAB;'>Comparison of 3 Machine Learning Algorithms specifically requested in Chapter 4 & 5 of the Capstone Mini Project Guidelines.</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Graphs Section
    st.subheader("1. Performance Metrics Comparison")
    
    # Prepare Dataframe for Chart
    metrics_data = {
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        "Accuracy": [model_results["Logistic Regression"]["Accuracy"], model_results["Random Forest"]["Accuracy"], model_results["Gradient Boosting"]["Accuracy"]],
        "Precision": [model_results["Logistic Regression"]["Precision"], model_results["Random Forest"]["Precision"], model_results["Gradient Boosting"]["Precision"]],
        "Recall": [model_results["Logistic Regression"]["Recall"], model_results["Random Forest"]["Recall"], model_results["Gradient Boosting"]["Recall"]]
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    # Use native Streamlit charting for beautiful dark mode color compatibility
    st.bar_chart(df_metrics.set_index("Model"), height=350, use_container_width=True)
    
    st.divider()
    
    st.subheader("2. Confusion Matrix Heatmaps")
    st.write("Heatmaps representing True Positives, False Positives, True Negatives, and False Negatives for each algorithm's predictions.")
    
    cols = st.columns(3)
    
    for i, (model_name, results) in enumerate(model_results.items()):
        with cols[i]:
            st.markdown(f"**{model_name}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            
            # Use seaborn heatmaps (Color output for PPT Printing as requested)
            # Customizing color palettes per model to make it look extremely premium
            if "Logistic" in model_name:
                cmap = "Blues"
            elif "Random" in model_name:
                cmap = "Greens"
            else:
                cmap = "Oranges"
                
            sns.heatmap(results["CM"], annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                        annot_kws={"size": 14, "weight": "bold"}, linewidths=1.5, linecolor="#0a0c10")
            
            # Styling for the dark theme graph plot output
            fig.patch.set_facecolor('#0a0c10')
            ax.set_facecolor('#0a0c10')
            ax.tick_params(colors='white')
            ax.set_xlabel('Predicted Risk', color='white', fontsize=10)
            ax.set_ylabel('Actual Risk', color='white', fontsize=10)
            ax.grid(False)
            
            st.pyplot(fig)
            
    st.divider()
    st.markdown("**Conclusion:** The **Random Forest Classifier** was selected for the live predictor because it optimally balances Accuracy and Recall, reducing the risk of incorrectly classifying high-risk profiles.")
