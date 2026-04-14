import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, confusion_matrix, roc_curve)

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

st.markdown("""
<style>
.risk-high   { background:#ffe0e0; padding:20px; border-radius:10px; border-left:5px solid #e74c3c; }
.risk-medium { background:#fff3cd; padding:20px; border-radius:10px; border-left:5px solid #f39c12; }
.risk-low    { background:#d4edda; padding:20px; border-radius:10px; border-left:5px solid #27ae60; }
</style>""", unsafe_allow_html=True)

st.markdown("## 📊 Customer Churn Predictor")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df = df.drop(columns=['customerID'])
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    df['AvgMonthlyCharge']   = df['TotalCharges'] / (df['tenure'] + 1)
    df['ServiceCount']       = df[['OnlineSecurity','OnlineBackup','DeviceProtection',
                                    'TechSupport','StreamingTV','StreamingMovies']].apply(
                                    lambda x: (x == 'Yes').sum(), axis=1)
    df['IsNewCustomer']      = (df['tenure'] <= 6).astype(int)
    df['IsLongTermCustomer'] = (df['tenure'] > 24).astype(int)
    df['HighCharges']        = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','MultipleLines']
    ohe_cols    = ['InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
    num_cols    = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen',
                   'AvgMonthlyCharge','ServiceCount','IsNewCustomer','IsLongTermCustomer','HighCharges']

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols + binary_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), ohe_cols)
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall'   : recall_score(y_test, y_pred),
        'F1 Score' : f1_score(y_test, y_pred),
        'ROC-AUC'  : roc_auc_score(y_test, y_proba)
    }

    return pipe, df, X_test, y_test, metrics

with st.spinner("⏳ Model train ho raha hai... thoda wait karo (1-2 min)"):
    try:
        model, df, X_test, y_test, metrics = load_and_train()
        st.success("✅ Model ready hai!")
    except FileNotFoundError:
        st.error("❌ CSV file nahi mili! app.py ke saath wali folder me CSV rakho.")
        st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Dataset", "📈 Performance"])

# ── TAB 1 ─────────────────────────────────────────────────────
with tab1:
    st.header("🔮 Customer Ka Churn Risk Check Karo")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal")
        gender  = st.selectbox("Gender", ["Male", "Female"])
        senior  = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        deps    = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        st.subheader("📱 Services")
        tenure      = st.slider("Tenure (Months)", 0, 72, 12)
        phone_svc   = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_sec  = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_bkp  = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        dev_prot    = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_sup    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        stream_tv   = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        stream_mov  = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("💰 Billing")
        contract  = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment   = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total   = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly * tenure))

    if st.button("🔮 Predict Karo!", use_container_width=True, type="primary"):

        svc_count = sum([online_sec=="Yes", online_bkp=="Yes", dev_prot=="Yes",
                         tech_sup=="Yes", stream_tv=="Yes", stream_mov=="Yes"])

        ml_map = {"No":0, "Yes":1, "No phone service":2}

        inp = pd.DataFrame([{
            'gender'           : 1 if gender=="Male" else 0,
            'SeniorCitizen'    : 1 if senior=="Yes" else 0,
            'Partner'          : 1 if partner=="Yes" else 0,
            'Dependents'       : 1 if deps=="Yes" else 0,
            'tenure'           : tenure,
            'PhoneService'     : 1 if phone_svc=="Yes" else 0,
            'MultipleLines'    : ml_map.get(multi_lines, 0),
            'InternetService'  : internet,
            'OnlineSecurity'   : online_sec,
            'OnlineBackup'     : online_bkp,
            'DeviceProtection' : dev_prot,
            'TechSupport'      : tech_sup,
            'StreamingTV'      : stream_tv,
            'StreamingMovies'  : stream_mov,
            'Contract'         : contract,
            'PaperlessBilling' : 1 if paperless=="Yes" else 0,
            'PaymentMethod'    : payment,
            'MonthlyCharges'   : monthly,
            'TotalCharges'     : total,
            'AvgMonthlyCharge' : total / (tenure + 1),
            'ServiceCount'     : svc_count,
            'IsNewCustomer'    : 1 if tenure <= 6 else 0,
            'IsLongTermCustomer': 1 if tenure > 24 else 0,
            'HighCharges'      : 1 if monthly > 65 else 0,
        }])

        prob = model.predict_proba(inp)[0][1]
        pred = model.predict(inp)[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.1%}")
        c2.metric("Prediction", "⚠️ Churn" if pred==1 else "✅ No Churn")
        c3.metric("Confidence", f"{max(prob, 1-prob):.1%}")

        if prob >= 0.7:
            st.markdown(f'<div class="risk-high"><h3>🔴 HIGH RISK — {prob:.1%}</h3>Turant retention offer do!</div>', unsafe_allow_html=True)
        elif prob >= 0.4:
            st.markdown(f'<div class="risk-medium"><h3>🟡 MEDIUM RISK — {prob:.1%}</h3>Discount ya upgrade offer karo.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low"><h3>🟢 LOW RISK — {prob:.1%}</h3>Customer loyal hai!</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 1.8))
        ax.barh(0, 0.4, color="#27ae60", alpha=0.7, height=0.5, label="Low")
        ax.barh(0, 0.3, left=0.4, color="#f39c12", alpha=0.7, height=0.5, label="Medium")
        ax.barh(0, 0.3, left=0.7, color="#e74c3c", alpha=0.7, height=0.5, label="High")
        ax.axvline(x=prob, color="black", linewidth=3, label=f"{prob:.1%}")
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xlabel("Churn Probability"); ax.legend(loc="upper left", fontsize=9)
        st.pyplot(fig); plt.close()

# ── TAB 2 ─────────────────────────────────────────────────────
with tab2:
    st.header("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churned", f"{df['Churn'].sum():,}")
    c3.metric("Churn Rate", f"{df['Churn'].mean():.1%}")

    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4, 4))
        vals = df['Churn'].value_counts()
        ax.pie(vals, labels=["No Churn","Churn"], autopct='%1.1f%%',
               colors=['#27ae60','#e74c3c'], startangle=90)
        ax.set_title("Churn Distribution")
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        df[df['Churn']==0]['tenure'].hist(bins=20, alpha=0.7, color='#27ae60', label='No Churn', ax=ax)
        df[df['Churn']==1]['tenure'].hist(bins=20, alpha=0.7, color='#e74c3c', label='Churn', ax=ax)
        ax.set_xlabel("Tenure (Months)"); ax.legend(); ax.set_title("Tenure Distribution")
        st.pyplot(fig); plt.close()

# ── TAB 3 ─────────────────────────────────────────────────────
with tab3:
    st.header("📈 Model Performance")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics['Accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['Precision']:.2%}")
    c3.metric("Recall",    f"{metrics['Recall']:.2%}")
    c4.metric("F1 Score",  f"{metrics['F1 Score']:.2%}")
    c5.metric("ROC-AUC",   f"{metrics['ROC-AUC']:.4f}")

    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test  = model.predict(X_test)
    fpr, tpr, _  = roc_curve(y_test, y_proba_test)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#1F4E79', linewidth=2, label=f"AUC = {metrics['ROC-AUC']:.3f}")
        ax.plot([0,1],[0,1],'k--', alpha=0.5)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend()
        st.pyplot(fig); plt.close()

    with col2:
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'])
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig); plt.close()

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Telco Churn ML Project ❤️</p>", unsafe_allow_html=True)
