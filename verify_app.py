import sys
import os

# Add local path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_processor import generate_simulated_data, validate_data, clean_data, engineer_features
from utils.models import train_logistic_regression, predict_single_customer, forecast_renewal_revenue_impact, forecast_future_trends
from utils.report_generator import generate_pdf_report, generate_excel_report

def run_verification():
    print("--------------------------------------------------")
    print("🚀 Starting Predictive Churn Platform Verification")
    print("--------------------------------------------------")
    
    # 1. Test Data Generation
    print("🧪 1. Testing Data Generation...")
    df = generate_simulated_data(num_records=500, random_seed=42)
    print(f"   Success: Generated dataset with shape {df.shape}")
    
    # 2. Test Data Validation
    print("🧪 2. Testing Data Validation...")
    val = validate_data(df)
    print(f"   Success: Found {val['total_missing']} missing values, {val['duplicate_count']} duplicates, {val['total_outliers']} outliers.")
    
    # 3. Test Data Cleaning
    print("🧪 3. Testing Data Cleaning & Feature Engineering...")
    df_cleaned, log = clean_data(df, missing_strategy='Mean Imputation', outlier_method='IQR', outlier_threshold=1.5, remove_duplicates=True)
    df_engineered = engineer_features(df_cleaned)
    print(f"   Success: Cleaned data shape: {df_cleaned.shape}. Engineered data shape: {df_engineered.shape}")
    print(f"   Engineered columns: {[c for c in df_engineered.columns if c not in df.columns]}")
    
    # 4. Test Model Training
    print("🧪 4. Testing Logistic Regression Training...")
    features = ['Age', 'Tenure', 'Monthly_Spend', 'Login_Frequency', 'Session_Duration', 'Support_Tickets', 'Product_Usage', 'Last_Login_Days']
    results = train_logistic_regression(df_engineered, feature_cols=features, target_col='Renewal_Status', train_size=0.8)
    print(f"   Success: Trained model. Test Accuracy: {results['metrics']['accuracy']*100:.2f}%")
    print(f"   ROC-AUC Score: {results['metrics']['roc_auc']:.3f}")
    
    # 5. Test Scoring & Revenue Impact
    print("🧪 5. Testing Scoring and MRR Retention Impacts...")
    impact = forecast_renewal_revenue_impact(df_engineered, results['model'], results['scaler'], features)
    print(f"   Success: Expected Renewals: {impact['expected_renewals']} / {impact['total_customers']}")
    print(f"   MRR Retained: ${impact['expected_revenue_retained']:,.2f} | MRR At Risk: ${impact['expected_revenue_at_risk']:,.2f}")
    
    # 6. Test Single Prediction
    print("🧪 6. Testing Individual Risk Prediction...")
    single_cust = {
        'Age': 35,
        'Tenure': 12,
        'Monthly_Spend': 49.99,
        'Login_Frequency': 25,
        'Session_Duration': 45.0,
        'Support_Tickets': 1,
        'Product_Usage': 85,
        'Last_Login_Days': 2
    }
    pred_res = predict_single_customer(results['model'], results['scaler'], features, single_cust)
    print(f"   Success: Single client renewal probability: {pred_res['probability']*100:.2f}%. Tier: {pred_res['risk_level']}")
    
    # 7. Test Trend Forecasting
    print("🧪 7. Testing Cohort Trend Linear Regression...")
    trend = forecast_future_trends(df_engineered, num_months=6)
    print(f"   Success: Next Month Forecasted renewal rate: {trend['next_month_rate']*100:.2f}%")
    
    # 8. Test Reports
    print("🧪 8. Testing Executive Report Exports (PDF and Excel)...")
    pdf_path = "test_executive_report.pdf"
    excel_path = "test_customer_database.xlsx"
    
    risk_counts = impact['scored_df']['Risk_Level'].value_counts().to_dict()
    generate_pdf_report(results['metrics'], results['coef_df'], risk_counts, impact, pdf_path)
    generate_excel_report(impact['scored_df'], excel_path)
    
    # Check if files were created
    if os.path.exists(pdf_path) and os.path.exists(excel_path):
        print(f"   Success: Report files created successfully.")
        # Cleanup
        os.remove(pdf_path)
        os.remove(excel_path)
        print("   Success: Temporary report files cleaned up.")
    else:
        print("   ❌ Error: Report files were not created.")
        sys.exit(1)
        
    print("--------------------------------------------------")
    print("🎉 Verification Complete! All systems operational.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_verification()
