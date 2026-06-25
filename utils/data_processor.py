import pandas as pd
import numpy as np
import io

def generate_simulated_data(num_records=1000, random_seed=42):
    """
    Generates a synthetic customer dataset with realistic relationships and
    intentionally introduces duplicates, missing values, and outliers
    for demonstrating cleaning capabilities.
    """
    np.random.seed(random_seed)
    
    # Generate base features
    customer_ids = [f"C-{1000 + i}" for i in range(num_records)]
    age = np.random.randint(18, 71, size=num_records)
    tenure = np.random.randint(1, 73, size=num_records) # months
    monthly_spend = np.round(np.random.uniform(15, 150, size=num_records), 2)
    
    # Engagement metrics (related to tenure and age)
    login_frequency = np.random.poisson(lam=15, size=num_records) + np.random.randint(0, 10, size=num_records)
    session_duration = np.round(np.random.normal(loc=30, scale=12, size=num_records) + login_frequency * 0.5, 1)
    session_duration = np.clip(session_duration, 1, 180)
    
    support_tickets = np.random.poisson(lam=1.5, size=num_records)
    # Increase tickets for users with high age or random issues
    support_tickets = np.clip(support_tickets, 0, 15)
    
    product_usage = np.random.randint(5, 100, size=num_records)
    last_login_days = np.random.geometric(p=0.08, size=num_records) - 1
    last_login_days = np.clip(last_login_days, 0, 60)
    
    subscription_type = np.random.choice(['Basic', 'Premium'], size=num_records, p=[0.6, 0.4])
    
    # Establish renewal relationship (higher logins, tenure, and premium subscription increase renewal)
    # higher support tickets and days since last login decrease renewal
    sub_type_numeric = np.where(subscription_type == 'Premium', 1.0, 0.0)
    
    # Logit calculation
    logit_prob = (
        -1.8 
        + 0.04 * tenure 
        + 0.08 * login_frequency 
        + 0.02 * session_duration 
        + 0.015 * product_usage 
        - 0.5 * support_tickets 
        - 0.09 * last_login_days 
        + 0.6 * sub_type_numeric
    )
    
    # Sigmoid function for probability
    prob = 1 / (1 + np.exp(-logit_prob))
    
    # Renewal status target variable (Bernoulli trials)
    renewal_status = np.random.binomial(n=1, p=prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': age,
        'Tenure': tenure,
        'Monthly_Spend': monthly_spend,
        'Login_Frequency': login_frequency,
        'Session_Duration': session_duration,
        'Support_Tickets': support_tickets,
        'Product_Usage': product_usage,
        'Last_Login_Days': last_login_days,
        'Subscription_Type': subscription_type,
        'Renewal_Status': renewal_status
    })
    
    # Let's introduce missing values (NaN) in ~5% of records for some features
    nan_mask_login = np.random.rand(num_records) < 0.04
    df.loc[nan_mask_login, 'Login_Frequency'] = np.nan
    
    nan_mask_spend = np.random.rand(num_records) < 0.03
    df.loc[nan_mask_spend, 'Monthly_Spend'] = np.nan
    
    nan_mask_sub = np.random.rand(num_records) < 0.02
    df.loc[nan_mask_sub, 'Subscription_Type'] = np.nan
    
    # Introduce duplicates (~3% of records duplicated)
    num_duplicates = int(num_records * 0.03)
    dup_indices = np.random.choice(num_records, size=num_duplicates, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    
    # Introduce outliers (explicit anomalies required by prompt)
    # Login Frequency = 5000
    # Session Duration = 0
    # Negative spends, etc.
    outlier_indices = np.random.choice(num_records, size=10, replace=False)
    
    df.loc[outlier_indices[0], 'Login_Frequency'] = 5000.0
    df.loc[outlier_indices[1], 'Session_Duration'] = 0.0
    df.loc[outlier_indices[2], 'Monthly_Spend'] = -99.0
    df.loc[outlier_indices[3], 'Age'] = 150 # Outlier age
    df.loc[outlier_indices[4], 'Last_Login_Days'] = 365 # Way beyond 60 days
    df.loc[outlier_indices[5], 'Tenure'] = -5 # Negative tenure
    
    # Append duplicates
    df = pd.concat([df, dup_df], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df

def parse_file(uploaded_file):
    """
    Parses CSV, XLSX, or JSON files uploaded through Streamlit
    and returns a Pandas DataFrame.
    """
    filename = uploaded_file.name
    if filename.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    elif filename.endswith('.json'):
        return pd.read_json(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV, Excel, or JSON.")

def check_outliers(df, column, method='IQR', threshold=1.5):
    """
    Identifies outliers in a column using IQR or Z-Score.
    Returns boolean Series where True indicates an outlier.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series(False, index=df.index)
        
    non_null_col = df[column].dropna()
    if len(non_null_col) == 0:
        return pd.Series(False, index=df.index)
        
    if method == 'IQR':
        q1 = non_null_col.quantile(0.25)
        q3 = non_null_col.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'Z-Score':
        mean = non_null_col.mean()
        std = non_null_col.std()
        if std == 0:
            return pd.Series(False, index=df.index)
        z_scores = (df[column] - mean) / std
        return z_scores.abs() > threshold
    return pd.Series(False, index=df.index)

def validate_data(df):
    """
    Analyzes dataset and returns validation metrics (missing values, duplicates, outliers).
    """
    total_records = len(df)
    
    # Check duplicates (based on Customer_ID if present, else entire row)
    if 'Customer_ID' in df.columns:
        duplicate_count = df.duplicated(subset=['Customer_ID']).sum()
    else:
        duplicate_count = df.duplicated().sum()
        
    # Check missing values
    missing_counts = df.isnull().sum().to_dict()
    total_missing = df.isnull().sum().sum()
    
    # Check outliers for key numeric fields
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter target status
    if 'Renewal_Status' in numeric_cols:
        numeric_cols.remove('Renewal_Status')
        
    outliers_info = {}
    total_outliers = 0
    for col in numeric_cols:
        # Default Z-score method with 3 std dev as initial validation filter
        outliers = check_outliers(df, col, method='Z-Score', threshold=3.0)
        cnt = outliers.sum()
        outliers_info[col] = int(cnt)
        total_outliers += cnt
        
    return {
        'total_records': total_records,
        'duplicate_count': int(duplicate_count),
        'missing_counts': missing_counts,
        'total_missing': int(total_missing),
        'outliers_by_col': outliers_info,
        'total_outliers': int(total_outliers)
    }

def clean_data(df, missing_strategy='Mean Imputation', outlier_method='IQR', outlier_threshold=1.5, remove_duplicates=True):
    """
    Cleans the dataset according to specified strategies.
    Returns cleaned DataFrame and cleaning log metrics.
    """
    df_clean = df.copy()
    log = {
        'initial_records': len(df),
        'duplicates_removed': 0,
        'missing_fixed': 0,
        'outliers_removed': 0
    }
    
    # 1. Handle Duplicates
    if remove_duplicates:
        initial_len = len(df_clean)
        if 'Customer_ID' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['Customer_ID'], keep='first')
        else:
            df_clean = df_clean.drop_duplicates(keep='first')
        log['duplicates_removed'] = initial_len - len(df_clean)
        
    # 2. Handle Outliers
    # Identify numeric columns for outlier detection
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'Renewal_Status' in numeric_cols:
        numeric_cols.remove('Renewal_Status')
        
    outlier_mask = pd.Series(False, index=df_clean.index)
    for col in numeric_cols:
        outliers = check_outliers(df_clean, col, method=outlier_method, threshold=outlier_threshold)
        outlier_mask = outlier_mask | outliers
        
    log['outliers_removed'] = outlier_mask.sum()
    df_clean = df_clean[~outlier_mask].reset_index(drop=True)
    
    # 3. Handle Missing Values
    # We impute or remove missing values in df_clean
    for col in df_clean.columns:
        null_count = df_clean[col].isnull().sum()
        if null_count > 0:
            log['missing_fixed'] += null_count
            if missing_strategy == 'Row Removal':
                df_clean = df_clean.dropna(subset=[col])
            else:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    if missing_strategy == 'Mean Imputation':
                        val = df_clean[col].mean()
                    elif missing_strategy == 'Median Imputation':
                        val = df_clean[col].median()
                    elif missing_strategy == 'Mode Imputation':
                        val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 0
                    df_clean[col] = df_clean[col].fillna(val)
                else:
                    # Categorical column
                    val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(val)
                    
    df_clean = df_clean.reset_index(drop=True)
    log['final_records'] = len(df_clean)
    
    return df_clean, log

def engineer_features(df):
    """
    Applies feature engineering formulas to customer behavioral data:
    1. Engagement_Score = Login_Frequency * Session_Duration
    2. Support_Risk = Support_Tickets / Tenure
    3. Recency_Score = Days Since Last Login (Last_Login_Days)
    4. Customer_Health_Score = scaled engagement, support, and recency
    """
    df_eng = df.copy()
    
    # 1. Engagement Score
    df_eng['Engagement_Score'] = df_eng['Login_Frequency'] * df_eng['Session_Duration']
    
    # 2. Support Risk (using Tenure + 1 to avoid division by zero or negative values)
    # Ensure Tenure is positive
    tenure_safe = np.where(df_eng['Tenure'] <= 0, 1, df_eng['Tenure'])
    df_eng['Support_Risk'] = np.round(df_eng['Support_Tickets'] / tenure_safe, 4)
    
    # 3. Recency Score (represent last login days directly as required)
    df_eng['Recency_Score'] = df_eng['Last_Login_Days']
    
    # 4. Customer Health Score (0 to 100 scale)
    # Scaled Engagement (Higher is better)
    eng_min = df_eng['Engagement_Score'].min()
    eng_max = df_eng['Engagement_Score'].max()
    eng_range = (eng_max - eng_min) if (eng_max - eng_min) > 0 else 1
    scaled_eng = 100.0 * (df_eng['Engagement_Score'] - eng_min) / eng_range
    
    # Scaled Support Risk (Lower is better)
    sup_min = df_eng['Support_Risk'].min()
    sup_max = df_eng['Support_Risk'].max()
    sup_range = (sup_max - sup_min) if (sup_max - sup_min) > 0 else 1
    scaled_sup = 100.0 * (df_eng['Support_Risk'] - sup_min) / sup_range
    
    # Scaled Recency Score / Last login days (Lower is better)
    rec_min = df_eng['Recency_Score'].min()
    rec_max = df_eng['Recency_Score'].max()
    rec_range = (rec_max - rec_min) if (rec_max - rec_min) > 0 else 1
    scaled_rec = 100.0 * (df_eng['Recency_Score'] - rec_min) / rec_range
    
    # Define health score:
    # 40% Engagement, 30% low Support Risk, 30% low Recency (Active recently)
    df_eng['Customer_Health_Score'] = np.round(
        0.4 * scaled_eng + 
        0.3 * (100.0 - scaled_sup) + 
        0.3 * (100.0 - scaled_rec), 
        1
    )
    
    return df_eng
