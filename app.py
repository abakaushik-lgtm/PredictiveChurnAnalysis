import streamlit as st
import pandas as pd
import numpy as np
import os
import io

# Import custom utilities
from utils.data_processor import (
    generate_simulated_data,
    parse_file,
    validate_data,
    clean_data,
    engineer_features
)
from utils.models import (
    train_logistic_regression,
    predict_single_customer,
    forecast_renewal_revenue_impact,
    forecast_future_trends
)
from utils.visualizations import (
    plot_churn_distribution,
    plot_behavioral_analysis,
    plot_session_duration_distribution,
    plot_subscription_plan_analysis,
    plot_tenure_vs_probability,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_risk_distribution,
    plot_forecast_trends
)
from utils.report_generator import (
    generate_pdf_report,
    generate_excel_report
)

# Page configuration
st.set_page_config(
    page_title="Predictive Churn & Retention Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and clean premium dashboard styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Title Banner */
    .banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
    }
    
    .banner h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.05em;
    }
    
    .banner p {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Styled KPI Card Container */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Individual KPI Card */
    .kpi-card {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-left: 5px solid #3B82F6;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .kpi-title {
        font-size: 0.875rem;
        color: #4B5563;
        font-weight: 500;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Risk KPI Colorings */
    .kpi-retained { border-left-color: #10B981; }
    .kpi-risk { border-left-color: #EF4444; }
    .kpi-warning { border-left-color: #F59E0B; }
    
    /* Status Badge styling */
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .badge-yellow { background-color: #FEF3C7; color: #92400E; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }
    
    /* Sidebar styling enhancements */
    .sidebar-section {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if 'raw_df' not in st.session_state:
    st.session_state['raw_df'] = None
if 'cleaned_df' not in st.session_state:
    st.session_state['cleaned_df'] = None
if 'engineered_df' not in st.session_state:
    st.session_state['engineered_df'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'scored_df' not in st.session_state:
    st.session_state['scored_df'] = None
if 'revenue_impact' not in st.session_state:
    st.session_state['revenue_impact'] = None
if 'cleaning_log' not in st.session_state:
    st.session_state['cleaning_log'] = None
if 'trend_results' not in st.session_state:
    st.session_state['trend_results'] = None

# Title Banner
st.markdown("""
<div class="banner">
    <h1>Predictive Churn & Retention Analytics Platform</h1>
    <p>Upload subscription data, execute advanced data cleaning, train machine learning models, segment customer risk, and estimate retention revenue impact.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🛠️ Configuration Panel")

# Section 1: Ingestion
st.sidebar.subheader("1. Data Ingestion")
data_source = st.sidebar.radio("Data Ingest Mode", ["Simulate Demo Data", "Upload Custom File"])

if data_source == "Simulate Demo Data":
    sim_size = st.sidebar.slider("Number of simulated records", min_value=200, max_value=5000, value=1500, step=100)
    if st.sidebar.button("Generate & Load Demo Data", use_container_width=True) or st.session_state['raw_df'] is None:
        with st.spinner("Generating simulated customer data..."):
            df = generate_simulated_data(num_records=sim_size, random_seed=42)
            st.session_state['raw_df'] = df
            # Reset subsequent states
            st.session_state['cleaned_df'] = None
            st.session_state['engineered_df'] = None
            st.session_state['model_results'] = None
            st.session_state['scored_df'] = None
            st.session_state['revenue_impact'] = None
            st.success("Simulated dataset loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel or JSON", type=['csv', 'xlsx', 'xls', 'json'])
    if uploaded_file is not None:
        try:
            df = parse_file(uploaded_file)
            st.session_state['raw_df'] = df
            # Reset subsequent states
            st.session_state['cleaned_df'] = None
            st.session_state['engineered_df'] = None
            st.session_state['model_results'] = None
            st.session_state['scored_df'] = None
            st.session_state['revenue_impact'] = None
            st.success("Uploaded file parsed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")

# Ensure we have data loaded before displaying tabs
if st.session_state['raw_df'] is None:
    st.info("👋 Welcome! Please click 'Generate & Load Demo Data' in the sidebar or upload your customer dataset to get started.")
    st.stop()

df_current = st.session_state['raw_df']

# Section 2: Model Setting quick controls
st.sidebar.subheader("2. Model Settings")
split_ratio = st.sidebar.slider("Train Split Ratio (%)", min_value=50, max_value=90, value=80) / 100.0
reg_penalty = st.sidebar.selectbox("Regularization Penalty", ["l2", "l1"])
c_value = st.sidebar.slider("Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.05)

# Quick Sidebar Summary if model is trained
st.sidebar.subheader("3. Platform Status")
if st.session_state['model_results'] is not None:
    acc = st.session_state['model_results']['metrics']['accuracy']
    st.sidebar.metric(label="Model Accuracy", value=f"{acc * 100:.1f}%")
    
    if st.session_state['revenue_impact'] is not None:
        rr = st.session_state['revenue_impact']['expected_renewal_rate'] * 100
        st.sidebar.metric(label="Expected Renewal Rate", value=f"{rr:.1f}%")
else:
    st.sidebar.info("Model not trained yet. Navigate to 'Churn Prediction' to train.")

# ----------------- TABS SETUP -----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 Dataset Overview",
    "🧹 Data Cleaning",
    "📊 Behavioral Analytics",
    "🤖 Churn Prediction",
    "⚠️ Risk Assessment",
    "📈 Forecast & Revenue Impact",
    "📄 Reports"
])

# ----------------- TAB 1: DATASET OVERVIEW -----------------
with tab1:
    st.header("📂 Subscription Dataset Ingestion & Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Table Preview")
        st.dataframe(df_current.head(100), height=350, use_container_width=True)
        
    with col2:
        st.subheader("Dataset Structure & Statistics")
        validation = validate_data(df_current)
        
        st.markdown(f"**Shape**: `{df_current.shape[0]:,}` rows × `{df_current.shape[1]}` columns")
        st.markdown(f"**Missing Values Total**: `{validation['total_missing']:,}` cells")
        st.markdown(f"**Duplicate Rows**: `{validation['duplicate_count']:,}` rows")
        st.markdown(f"**Potential Outliers (Z > 3)**: `{validation['total_outliers']:,}` data points")
        
        # Display data types
        dtypes_df = pd.DataFrame(df_current.dtypes, columns=['DataType']).astype(str)
        st.dataframe(dtypes_df, use_container_width=True)
        
    st.divider()
    
    st.subheader("Numeric Distributions Summary")
    st.dataframe(df_current.describe().T.round(2), use_container_width=True)


# ----------------- TAB 2: DATA CLEANING -----------------
with tab2:
    st.header("🧹 Data Cleaning & Validation Dashboard")
    
    st.markdown("""
    Automatically process issues such as missing data, duplicates, and outliers. 
    Select cleaning criteria below and apply changes.
    """)
    
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        missing_strategy = st.selectbox(
            "Missing Value Strategy",
            ["Mean Imputation", "Median Imputation", "Mode Imputation", "Row Removal"]
        )
    with cc2:
        outlier_method = st.selectbox("Outlier Detection Method", ["IQR", "Z-Score"])
    with cc3:
        outlier_thresh = st.slider("Outlier Threshold Multiplier", min_value=1.0, max_value=4.0, value=1.5 if outlier_method == 'IQR' else 3.0, step=0.1)
        
    remove_dups = st.checkbox("Remove Duplicate Records (Customer_ID or Row level)", value=True)
    
    if st.button("Apply Selected Cleaning Steps", type="primary", use_container_width=True):
        with st.spinner("Executing cleaning pipelines..."):
            cleaned_df, log = clean_data(
                df_current,
                missing_strategy=missing_strategy,
                outlier_method=outlier_method,
                outlier_threshold=outlier_thresh,
                remove_duplicates=remove_dups
            )
            
            # Apply feature engineering on clean dataset
            engineered_df = engineer_features(cleaned_df)
            
            # Save to session state
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['engineered_df'] = engineered_df
            st.session_state['cleaning_log'] = log
            
            # Reset model states when data changes
            st.session_state['model_results'] = None
            st.session_state['scored_df'] = None
            st.session_state['revenue_impact'] = None
            
            st.success("Cleaning successfully executed!")
            
    # Check if cleaned data exists to show validation dashboard
    if st.session_state['cleaned_df'] is not None:
        log = st.session_state['cleaning_log']
        
        st.subheader("Data Validation Dashboard")
        
        # Draw metric cards for cleaning results
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">Initial Records Loaded</div>
                <div class="kpi-value">{log['initial_records']:,}</div>
            </div>
            <div class="kpi-card kpi-warning">
                <div class="kpi-title">Duplicates Removed</div>
                <div class="kpi-value">{log['duplicates_removed']:,}</div>
            </div>
            <div class="kpi-card kpi-warning">
                <div class="kpi-title">Missing Values Fixed</div>
                <div class="kpi-value">{log['missing_fixed']:,}</div>
            </div>
            <div class="kpi-card kpi-risk">
                <div class="kpi-title">Outliers Treated</div>
                <div class="kpi-value">{log['outliers_removed']:,}</div>
            </div>
            <div class="kpi-card kpi-retained">
                <div class="kpi-title">Cleaned Records Remaining</div>
                <div class="kpi-value">{log['final_records']:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display sample comparison
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            st.subheader("Original Data Sample")
            st.dataframe(df_current.head(5), use_container_width=True)
        with col_comp2:
            st.subheader("Cleaned & Engineered Data Sample")
            st.dataframe(st.session_state['engineered_df'].head(5), use_container_width=True)
            
        st.info("💡 Engine completed! In addition to cleaning, the following behavioral indicators have been engineered: Engagement_Score, Support_Risk, Recency_Score, Customer_Health_Score.")
    else:
        st.warning("⚠️ Data has not been cleaned yet. Configure parameters above and click 'Apply Selected Cleaning Steps'.")


# ----------------- TAB 3: BEHAVIORAL ANALYTICS -----------------
with tab3:
    st.header("📊 Behavioral Analytics & Exploratory Data Analysis")
    
    # Use cleaned data if available, otherwise raw
    analysis_df = st.session_state['engineered_df'] if st.session_state['engineered_df'] is not None else df_current
    
    if st.session_state['engineered_df'] is None:
        st.warning("⚠️ Showing statistics based on raw, uncleaned data. For accurate metrics, run Data Cleaning first.")
        
    if 'Renewal_Status' not in analysis_df.columns:
        st.error("No 'Renewal_Status' column in dataset. Ensure your target variable is named 'Renewal_Status'.")
        st.stop()
        
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(plot_churn_distribution(analysis_df), use_container_width=True)
        
    with row1_col2:
        # Plan analysis
        if 'Subscription_Type' in analysis_df.columns:
            st.plotly_chart(plot_subscription_plan_analysis(analysis_df), use_container_width=True)
        else:
            st.info("Subscription_Type column missing; subscription plan analysis skipped.")
            
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if 'Login_Frequency' in analysis_df.columns:
            st.plotly_chart(plot_behavioral_analysis(analysis_df, numeric_col='Login_Frequency'), use_container_width=True)
        else:
            st.info("Login_Frequency column missing; logins vs renewal analysis skipped.")
            
    with row2_col2:
        if 'Session_Duration' in analysis_df.columns:
            st.plotly_chart(plot_session_duration_distribution(analysis_df), use_container_width=True)
        else:
            st.info("Session_Duration column missing; session duration analysis skipped.")
            
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        if 'Tenure' in analysis_df.columns:
            # We mock the renewal probability in simple cohort for EDA
            analysis_df_sc = analysis_df.copy()
            analysis_df_sc['Renewal_Probability'] = analysis_df_sc['Renewal_Status'] # Use actual status as probability base
            st.plotly_chart(plot_tenure_vs_probability(analysis_df_sc), use_container_width=True)
        else:
            st.info("Tenure column missing; tenure analysis skipped.")
    with row3_col2:
        if 'Customer_Health_Score' in analysis_df.columns:
            st.subheader("Customer Health Score Distribution")
            fig_health = px.histogram(
                analysis_df,
                x='Customer_Health_Score',
                color='Renewal_Status',
                barmode='overlay',
                color_discrete_map={1: '#10B981', 0: '#EF4444'},
                title="Customer Health Score by Renewal Status (1=Renewed, 0=Churned)"
            )
            fig_health.update_layout(template="plotly_white")
            st.plotly_chart(fig_health, use_container_width=True)


# ----------------- TAB 4: CHURN PREDICTION -----------------
with tab4:
    st.header("🤖 Logistic Regression Predictive Modeling")
    
    # Require cleaned/engineered data
    if st.session_state['engineered_df'] is None:
        st.warning("⚠️ Please execute Data Cleaning (Tab 2) before training the predictive model.")
        st.stop()
        
    eng_df = st.session_state['engineered_df']
    
    st.markdown("Select features to train the **Logistic Regression** model:")
    
    # Identify available feature columns
    all_potential_features = [
        'Age', 'Tenure', 'Monthly_Spend', 'Login_Frequency', 'Session_Duration',
        'Support_Tickets', 'Product_Usage', 'Last_Login_Days', 'Subscription_Type_Encoded',
        'Engagement_Score', 'Support_Risk', 'Recency_Score'
    ]
    
    # Exclude technical or missing columns, keep existing ones
    # Ensure Subscription_Type is mapped to numeric
    if 'Subscription_Type' in eng_df.columns and 'Subscription_Type_Encoded' not in eng_df.columns:
        eng_df['Subscription_Type_Encoded'] = np.where(eng_df['Subscription_Type'] == 'Premium', 1, 0)
        st.session_state['engineered_df'] = eng_df
        
    available_features = [f for f in all_potential_features if f in eng_df.columns]
    
    # User selects columns
    selected_features = st.multiselect(
        "Model Predictor Variables",
        available_features,
        default=[f for f in available_features if f not in ['Subscription_Type_Encoded', 'Subscription_Type', 'Engagement_Score', 'Support_Risk', 'Recency_Score']]
    )
    
    if len(selected_features) == 0:
        st.warning("⚠️ Please select at least one feature to train the model.")
        st.stop()
        
    if st.button("Train Predictive Churn Model", type="primary", use_container_width=True):
        with st.spinner("Standardizing features and training logistic regression..."):
            results = train_logistic_regression(
                eng_df,
                feature_cols=selected_features,
                target_col='Renewal_Status',
                train_size=split_ratio,
                penalty=reg_penalty,
                C=c_value
            )
            st.session_state['model_results'] = results
            
            # Score the full dataset
            rev_impact = forecast_renewal_revenue_impact(
                eng_df,
                results['model'],
                results['scaler'],
                selected_features,
                monthly_spend_col='Monthly_Spend'
            )
            st.session_state['scored_df'] = rev_impact['scored_df']
            st.session_state['revenue_impact'] = rev_impact
            
            # Forecast trends
            trend_res = forecast_future_trends(eng_df, num_months=6)
            st.session_state['trend_results'] = trend_res
            
            st.success("Model trained and dataset scored successfully!")
            
    # Display trained model performance
    if st.session_state['model_results'] is not None:
        res = st.session_state['model_results']
        metrics_dict = res['metrics']
        
        st.subheader("Model Classification Evaluation")
        
        # Metric Cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Accuracy", f"{metrics_dict['accuracy']*100:.1f}%")
        col_m2.metric("Precision", f"{metrics_dict['precision']*100:.1f}%")
        col_m3.metric("Recall", f"{metrics_dict['recall']*100:.1f}%")
        col_m4.metric("F1 Score", f"{metrics_dict['f1_score']*100:.1f}%")
        col_m5.metric("ROC-AUC", f"{metrics_dict['roc_auc']:.3f}")
        
        st.divider()
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.plotly_chart(plot_roc_curve(
                res['roc_curve']['fpr'], 
                res['roc_curve']['tpr'], 
                metrics_dict['roc_auc']
            ), use_container_width=True)
            
        with col_plot2:
            st.plotly_chart(plot_feature_importance(res['coef_df']), use_container_width=True)
            
        st.divider()
        
        st.subheader("Model Statistical Summary")
        with st.expander("View Statsmodels Logit Summary Table (p-values & standard errors)"):
            st.markdown(res['statsmodels_summary_html'], unsafe_allow_html=True)
    else:
        st.info("ℹ️ Model has not been trained yet. Click 'Train Predictive Churn Model' above to view evaluation metrics.")


# ----------------- TAB 5: RISK ASSESSMENT -----------------
with tab5:
    st.header("⚠️ Customer Risk Assessment & Scorecard")
    
    # Require model results
    if st.session_state['model_results'] is not None and st.session_state['scored_df'] is not None:
        scored_df = st.session_state['scored_df']
        
        col_rc1, col_rc2 = st.columns([1, 2])
        
        with col_rc1:
            st.plotly_chart(plot_risk_distribution(scored_df), use_container_width=True)
            
        with col_rc2:
            st.subheader("High Risk Customers 🔴")
            high_risk_df = scored_df[scored_df['Risk_Level'] == 'High Churn Risk 🔴']
            cols_to_show = ['Customer_ID', 'Age', 'Tenure', 'Monthly_Spend', 'Support_Tickets', 'Customer_Health_Score', 'Renewal_Probability']
            st.dataframe(
                high_risk_df[cols_to_show].sort_values(by='Renewal_Probability').head(100), 
                use_container_width=True,
                height=300
            )
            
        st.divider()
        
        # Interactive Single Predictor
        st.subheader("🔮 Interactive Customer Prediction Tool")
        st.write("Input manual customer characteristics to predict individual renewal likelihood and risk tiers.")
        
        form_col1, form_col2, form_col3 = st.columns(3)
        res_model = st.session_state['model_results']
        model = res_model['model']
        scaler = res_model['scaler']
        features_trained = res_model['feature_cols']
        
        # Draw inputs depending on feature set
        with form_col1:
            in_tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12) if 'Tenure' in features_trained else None
            in_logins = st.number_input("Monthly Logins", min_value=0, max_value=1000, value=25) if 'Login_Frequency' in features_trained else None
            in_duration = st.number_input("Session Duration (Mins)", min_value=0.0, max_value=1440.0, value=40.0) if 'Session_Duration' in features_trained else None
        
        with form_col2:
            in_spend = st.number_input("Monthly Spend ($)", min_value=0.0, max_value=1000.0, value=49.99) if 'Monthly_Spend' in features_trained else None
            in_tickets = st.number_input("Support Tickets", min_value=0, max_value=100, value=1) if 'Support_Tickets' in features_trained else None
            in_age = st.number_input("Age", min_value=18, max_value=120, value=35) if 'Age' in features_trained else None
            
        with form_col3:
            in_usage = st.number_input("Product Usage Score (0-100)", min_value=0, max_value=100, value=80) if 'Product_Usage' in features_trained else None
            in_recency = st.number_input("Days Since Last Login", min_value=0, max_value=365, value=3) if 'Last_Login_Days' in features_trained else None
            in_sub = st.selectbox("Subscription Plan", ["Basic", "Premium"]) if ('Subscription_Type_Encoded' in features_trained or 'Subscription_Type' in features_trained) else None
            
        if st.button("Calculate Renewal Likelihood", type="primary", use_container_width=True):
            # Create a dictionary matching model features
            single_cust = {}
            if 'Tenure' in features_trained: single_cust['Tenure'] = in_tenure
            if 'Login_Frequency' in features_trained: single_cust['Login_Frequency'] = in_logins
            if 'Session_Duration' in features_trained: single_cust['Session_Duration'] = in_duration
            if 'Monthly_Spend' in features_trained: single_cust['Monthly_Spend'] = in_spend
            if 'Support_Tickets' in features_trained: single_cust['Support_Tickets'] = in_tickets
            if 'Age' in features_trained: single_cust['Age'] = in_age
            if 'Product_Usage' in features_trained: single_cust['Product_Usage'] = in_usage
            if 'Last_Login_Days' in features_trained: single_cust['Last_Login_Days'] = in_recency
            if 'Subscription_Type_Encoded' in features_trained: 
                single_cust['Subscription_Type_Encoded'] = 1.0 if in_sub == 'Premium' else 0.0
            
            # Engineered features helper logic (if model trained on them)
            # Recreate them dynamically
            if 'Engagement_Score' in features_trained:
                single_cust['Engagement_Score'] = in_logins * in_duration
            if 'Support_Risk' in features_trained:
                single_cust['Support_Risk'] = in_tickets / max(1, in_tenure)
            if 'Recency_Score' in features_trained:
                single_cust['Recency_Score'] = in_recency
                
            pred_res = predict_single_customer(model, scaler, features_trained, single_cust)
            prob_percent = pred_res['probability'] * 100.0
            
            # Draw beautiful result box
            st.markdown("### Prediction Result")
            
            if 'High Churn' in pred_res['risk_level']:
                card_class = "badge-red"
                emoji = "🔴"
                rec_list = [
                    "Offer immediate 20% loyalty discount code.",
                    "Assign primary customer success support agent.",
                    "Send personalized engagement guide."
                ]
            elif 'Medium' in pred_res['risk_level']:
                card_class = "badge-yellow"
                emoji = "🟡"
                rec_list = [
                    "Queue for upcoming re-engagement emails.",
                    "Evaluate if support tickets are marked unresolved.",
                    "Provide a free trial of premium features."
                ]
            else:
                card_class = "badge-green"
                emoji = "🟢"
                rec_list = [
                    "Flag for subscription upsell promotions.",
                    "Request product feedback or NPS review.",
                    "Invite to share referral links for rewards."
                ]
                
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1.5rem; border-radius: 8px;">
                    <h4>Renewal Probability</h4>
                    <h2 style="color: #1E3A8A; margin: 0;">{prob_percent:.1f}%</h2>
                    <p style="margin-top: 0.5rem; margin-bottom: 0;">Status: <b>{pred_res['prediction']}</b></p>
                    <p>Risk: <span class="badge {card_class}">{pred_res['risk_level']}</span></p>
                </div>
                """, unsafe_allow_html=True)
            with col_res2:
                st.markdown("#### Suggested Retention Actions")
                for rec in rec_list:
                    st.markdown(f"✓ {rec}")
    else:
        st.info("ℹ️ Churn prediction model must be trained to compute customer risks. Train the model under 'Churn Prediction' tab.")


# ----------------- TAB 6: FORECAST & REVENUE IMPACT -----------------
with tab6:
    st.header("📈 Financial Forecast & Churn Revenue Impact")
    
    if st.session_state['revenue_impact'] is not None and st.session_state['trend_results'] is not None:
        impact = st.session_state['revenue_impact']
        trend = st.session_state['trend_results']
        
        # Display KPI cards for revenue impact
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card kpi-retained">
                <div class="kpi-title">Expected Renewals</div>
                <div class="kpi-value">{impact['expected_renewals']:,}</div>
            </div>
            <div class="kpi-card kpi-risk">
                <div class="kpi-title">Expected Churn</div>
                <div class="kpi-value">{impact['expected_churn']:,}</div>
            </div>
            <div class="kpi-card kpi-retained">
                <div class="kpi-title">Projected Revenue Retained</div>
                <div class="kpi-value">${impact['expected_revenue_retained']:,.2f}</div>
            </div>
            <div class="kpi-card kpi-risk">
                <div class="kpi-title">Projected Revenue At Risk</div>
                <div class="kpi-value">${impact['expected_revenue_at_risk']:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("Linear Regression Forecasting (Cohort Trends)")
        st.markdown(f"""
        Using historical cohort indices, we fit a **Linear Regression model** to forecast subscription rates and expected revenue trends:
        - **Next Month Forecasted Renewal Rate**: `{trend['next_month_rate']*100:.1f}%`
        - **Next Month Forecasted Monthly Revenue**: `${trend['next_month_revenue']:,.2f}`
        """)
        
        col_tr1, col_tr2 = st.columns(2)
        with col_tr1:
            st.plotly_chart(plot_forecast_trends(trend['trend_df'], metric='Renewal_Rate'), use_container_width=True)
        with col_tr2:
            st.plotly_chart(plot_forecast_trends(trend['trend_df'], metric='Revenue'), use_container_width=True)
    else:
        st.info("ℹ️ Forecast calculations depend on the predictive model. Please train the model under the 'Churn Prediction' tab first.")


# ----------------- TAB 7: REPORTS -----------------
with tab7:
    st.header("📄 Export Business Intelligence & Executive Reports")
    
    if st.session_state['model_results'] is not None and st.session_state['scored_df'] is not None:
        st.write("Generate presentation-ready executive PDF summaries and formatted Microsoft Excel spreadsheets containing prediction tables.")
        
        rep_col1, rep_col2 = st.columns(2)
        
        with rep_col1:
            st.subheader("PDF Executive Report Settings")
            author_name = st.text_input("Author/Preparer Name", value="Predictive Churn Analytics Engine")
            report_title = st.text_input("Custom Document Header", value="Executive Subscription Retention Report")
            
            if st.button("Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Compiling ReportLab PDF document..."):
                    pdf_filename = "Executive_Churn_Report.pdf"
                    
                    res_m = st.session_state['model_results']
                    metrics_dict = res_m['metrics']
                    coef_df = res_m['coef_df']
                    scored_df = st.session_state['scored_df']
                    
                    risk_counts = scored_df['Risk_Level'].value_counts().to_dict()
                    rev_impact = st.session_state['revenue_impact']
                    
                    # Generate report bytes
                    generate_pdf_report(
                        metrics_dict=metrics_dict,
                        coef_df=coef_df,
                        risk_counts=risk_counts,
                        revenue_impact=rev_impact,
                        output_path=pdf_filename
                    )
                    
                    # Read generated PDF as bytes
                    with open(pdf_filename, "rb") as f:
                        pdf_bytes = f.read()
                        
                    st.download_button(
                        label="⬇️ Download Executive PDF Report",
                        data=pdf_bytes,
                        file_name="Executive_Customer_Churn_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    # Clean up local file
                    try:
                        os.remove(pdf_filename)
                    except:
                        pass
                        
        with rep_col2:
            st.subheader("Cleaned Dataset Spreadsheet Export")
            st.write("Export the complete customer table containing computed scores, predictions, and colored risk categorizations directly to Microsoft Excel.")
            
            if st.button("Generate Excel Spreadsheet", type="primary", use_container_width=True):
                with st.spinner("Structuring OpenPyXL Workbook columns..."):
                    xlsx_filename = "Scored_Customers_Database.xlsx"
                    scored_df = st.session_state['scored_df']
                    
                    generate_excel_report(scored_df, xlsx_filename)
                    
                    with open(xlsx_filename, "rb") as f:
                        xlsx_bytes = f.read()
                        
                    st.download_button(
                        label="⬇️ Download Scored Excel Dataset",
                        data=xlsx_bytes,
                        file_name="Customer_Retention_Scores.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # Clean up local file
                    try:
                        os.remove(xlsx_filename)
                    except:
                        pass
    else:
        st.info("ℹ️ Reports require a trained predictive model. Please navigate to 'Churn Prediction' to train the model first.")
