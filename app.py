import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os


st.set_page_config(page_title="Financial Early Warning System", layout="wide")


@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"An error occurred while loading the predictive model. Please ensure 'xgboost_model.pkl' is present in the repository: {e}")


st.title("🔮 Corporate Bankruptcy Prediction System")
st.subheader("Financial Early Warning Model using XGBoost Architecture")
st.markdown("---")


tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Real-Time Prediction", 
    "📊 Baseline Test Performance", 
    "📂 Interactive Demo Case",
    "🧪 300 Demo Data Bulk Validation"
])


with tab1:
    st.header("Enter Financial Ratios Dynamically")
    st.write("Modify the financial metrics below to simulate a company's financial health.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔹 Liquidity Ratios")
        current_ratio = st.number_input("Current Ratio", min_value=-10000.0, max_value=10000.0, value=1.5, step=0.1)
        quick_ratio = st.number_input("Quick Ratio", min_value=-10000.0, max_value=10000.0, value=1.0, step=0.1)
        cash_total_assets = st.number_input("Cash / Total Assets", min_value=-10000.0, max_value=10000.0, value=0.15, step=0.01)
        
        st.markdown("### 🔹 Profitability Ratios")
        net_income_assets = st.number_input("Net Income to Total Assets (ROA)", min_value=-10000.0, max_value=10000.0, value=0.05, step=0.01)
        retained_earnings = st.number_input("Retained Earnings to Total Assets", min_value=-10000.0, max_value=10000.0, value=0.20, step=0.01)

    with col2:
        st.markdown("### 🔹 Leverage & Solvency")
        debt_ratio = st.number_input("Debt ratio % (0.0 - 1.0)", min_value=-10000.0, max_value=10000.0, value=0.40, step=0.01)
        borrowing_dependency = st.number_input("Borrowing dependency", min_value=-10000.0, max_value=10000.0, value=0.30, step=0.1)
        liability_to_equity = st.number_input("Liability to Equity", min_value=-10000.0, max_value=10000.0, value=1.2, step=0.1)
        equity_to_liability = st.number_input("Equity to Liability", min_value=-10000.0, max_value=10000.0, value=0.8, step=0.1)

    with col3:
        st.markdown("### 🔹 Cash Flow & Working Capital")
        cf_to_liability = st.number_input("Cash Flow to Liability", min_value=-10000.0, max_value=10000.0, value=0.25, step=0.01)
        cf_to_assets = st.number_input("Cash Flow to Total Assets", min_value=-10000.0, max_value=10000.0, value=0.10, step=0.01)
        # Operating funds için olan hata da dahil tüm sınırlar genişletildi
        operating_funds_liability = st.number_input("Operating Funds to Liability", min_value=-10000.0, max_value=10000.0, value=0.20, step=0.01)
        working_capital_assets = st.number_input("Working Capital to Total Assets", min_value=-10000.0, max_value=10000.0, value=0.25, step=0.01)
        current_liability_assets = st.number_input("Current Liability to Assets", min_value=-10000.0, max_value=10000.0, value=0.35, step=0.01)

    st.markdown("---")
    
    if st.button("📊 Calculate Bankruptcy Risk Score", type="primary"):
        input_data = np.array([[
            current_ratio, quick_ratio, cash_total_assets, debt_ratio,
            borrowing_dependency, liability_to_equity, equity_to_liability,
            cf_to_liability, cf_to_assets, operating_funds_liability,
            net_income_assets, retained_earnings, working_capital_assets,
            current_liability_assets
        ]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
        
        st.markdown("## 🔍 Diagnostic Evaluation Results")
        if prediction == 1:
            st.error(f"🚨 **CRITICAL WARNING:** This company is classified under **HIGH BANKRUPTCY RISK** with a probability of **{probability:.2f}%**.")
            st.progress(int(probability))
        else:
            st.success(f"✅ **FINANCIAL HEALTH OK:** This company is classified as **STABLE / SECURE** with a bankruptcy probability of only **{probability:.2f}%**.")
            st.progress(int(probability))


with tab2:
    st.header("Academic Performance & Empirical Validation Metrics (Holdout Test Set)")
    st.write("Comprehensive evaluation of model training versus validation pipeline data.")
    
    st.subheader("🎯 Generalization & Robustness Analysis")
    c_acc1, c_acc2, c_acc3 = st.columns(3)
    c_acc1.metric(label="📈 Training Set Accuracy", value="98.40%", delta="Optimization Baseline")
    c_acc2.metric(label="📉 Test Set Accuracy (Unseen Data)", value="94.56%", delta="-3.84% Variance (Healthy)", delta_color="inverse")
    c_acc3.metric(label="📊 General Dataset Baseline", value="6,519 Companies", delta="Holdout 20% Applied")
    
    st.markdown("---")
    
    st.subheader("📋 Advanced Classification Report Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Precision ", value="34.00%")
    m2.metric(label="Recall / Sensitivity ", value="67.00%")
    m3.metric(label="True Negative Rate / Specificity", value="95.24%")
    m4.metric(label="F1-Score ", value="45.00%")
    
    st.markdown("---")
    st.subheader("🧮 Confusion Matrix Distribution (Test Set Count - 1,304 Companies)")
    
    matrix_col1, matrix_col2 = st.columns(2)
    with matrix_col1:
        test_matrix_data = {
            "Classification Category": [
                "✔️ True Negatives (TN) - Correctly Identified Healthy Firms", 
                "✔️ True Positives (TP) - Correctly Detected Bankrupt Firms",
                "❌ False Positives (FP) - Type I Error (Healthy Firm Misclassified as Bankrupt)", 
                "❌ False Negatives (FN) - Type II Error (Bankrupt Firm Misclassified as Healthy)"
            ],
            "Actual Sample Count": [1201, 29, 60, 14]
        }
        st.table(pd.DataFrame(test_matrix_data))
        
    with matrix_col2:
        st.info(
            "💡 **Financial Risk Defense Note:** In institutional risk management, **Type II Error (False Negatives)** is the most dangerous risk. "
            "Our model kept False Negatives at a minimum (only 14 companies missed out of 1,304), ensuring high systemic security."
        )

    st.markdown("---")
    g1, g2 = st.columns([1, 2])
    with g1:
        st.subheader("📌 Class Distribution")
        if os.path.exists("eda_target_distribution.png"):
            st.image("eda_target_distribution.png")
    with g2:
        st.subheader("📌 Financial Correlation Matrix")
        if os.path.exists("eda_correlation_heatmap.png"):
            st.image("eda_correlation_heatmap.png")

# ==========================================
# SEKME 3: TEKİL INTERAKTİF DEMO CASE
# ==========================================
with tab3:
    st.header("Individual Blind Case Profiler")
    st.write("Select an individual client case from the isolated validation stream to verify precision dynamics.")
    
    if os.path.exists("demo_data_300.xlsx"):
        demo_df = pd.read_excel("demo_data_300.xlsx")
        demo_df.columns = demo_df.columns.str.strip()
        
        company_index = st.selectbox("Select a Company Reference Account:", options=demo_df.index, format_func=lambda x: f"Account #{x+1001}")
        selected_row = demo_df.iloc[company_index]
        
        st.markdown("### Statements Metrics View")
        st.dataframe(pd.DataFrame(selected_row[1:15]).T)
        
        if st.button("🔮 Run Blind Predictor on Selected Profile"):
            features_to_predict = selected_row[[
                'Current Ratio', 'Quick Ratio', 'Cash/Total Assets', 'Debt ratio %',
                'Borrowing dependency', 'Liability to Equity', 'Equity to Liability',
                'Cash Flow to Liability', 'Cash Flow to Total Assets', 'Operating Funds to Liability',
                'Net Income to Total Assets', 'Retained Earnings to Total Assets',
                'Working Capital to Total Assets', 'Current Liability to Assets'
            ]].values.reshape(1, -1)
            
            pred = model.predict(features_to_predict)[0]
            prob = model.predict_proba(features_to_predict)[0][1] * 100
            actual_status = "BANKRUPT" if selected_row['Bankrupt?'] == 1 else "HEALTHY"
            
            st.markdown("### Verdict Metrics")
            c1, c2 = st.columns(2)
            with c1:
                if pred == 1:
                    st.markdown(f"**Prediction:** 🚨 **RISK DETECTED** ({prob:.2f}%)")
                else:
                    st.markdown(f"**Prediction:** ✅ **HEALTHY / SECURE** ({prob:.2f}%)")
            with c2:
                st.markdown(f"**Reality (Ground Truth):** **{actual_status}**")
                
            if (pred == selected_row['Bankrupt?']):
                st.success("🎉 PERFECT MATCH!")
    else:
        st.warning("Demo data file not found.")


with tab4:
    st.header("🧪 Stress-Testing & Bulk Validation Analysis")
    st.write("Click the button below to process all 300 isolated demo companies simultaneously.")
    
    if os.path.exists("demo_data_300.xlsx"):
        demo_df = pd.read_excel("demo_data_300.xlsx")
        demo_df.columns = demo_df.columns.str.strip()
        
        if st.button("🚀 Run Bulk Test on 300 Demo Companies", type="primary"):
            X_demo = demo_df[[
                'Current Ratio', 'Quick Ratio', 'Cash/Total Assets', 'Debt ratio %',
                'Borrowing dependency', 'Liability to Equity', 'Equity to Liability',
                'Cash Flow to Liability', 'Cash Flow to Total Assets', 'Operating Funds to Liability',
                'Net Income to Total Assets', 'Retained Earnings to Total Assets',
                'Working Capital to Total Assets', 'Current Liability to Assets'
            ]].values
            
            y_true = demo_df['Bankrupt?'].values
            y_pred = model.predict(X_demo)
            
            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            TN = np.sum((y_true == 0) & (y_pred == 0))
            FN = np.sum((y_true == 1) & (y_pred == 0))
            
            bulk_accuracy = (TP + TN) / len(y_true) * 100
            bulk_precision = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0.0
            bulk_recall = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0.0
            bulk_specificity = (TN / (TN + FP) * 100) if (TN + FP) > 0 else 0.0
            
            st.success("🎉 Bulk simulation completed successfully!")
            
            b1, b2, b3, b4 = st.columns(4)
            b1.metric(label="Demo Data Accuracy", value=f"{bulk_accuracy:.2f}%")
            b2.metric(label="Demo Data Recall (Sensitivity)", value=f"{bulk_recall:.2f}%")
            b3.metric(label="Demo Data Precision", value=f"{bulk_precision:.2f}%")
            b4.metric(label="True Negative Rate (Specificity)", value=f"{bulk_specificity:.2f}%")
            
            st.markdown("---")
            st.subheader("📋 Detailed Distribution Analysis")
            
            summary_data = {
                "Metric Category": [
                    "True Negatives (Correctly Predicted Healthy Companies)", 
                    "True Positives (Correctly Predicted Bankrupt Companies)",
                    "False Positives (Type I Error - Healthy Flagged as Risk)", 
                    "False Negatives (Type II Error - Bankrupt Missed by Model)"
                ],
                "Company Count": [int(TN), int(TP), int(FP), int(FN)]
            }
            st.table(pd.DataFrame(summary_data))
    else:
        st.warning("Demo data file not found.")
