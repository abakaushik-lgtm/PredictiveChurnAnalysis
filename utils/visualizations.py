import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Premium Plotly template settings
PLOTLY_TEMPLATE = "plotly_white"
COLOR_PRIMARY = "#3B82F6"   # Vibrant Blue
COLOR_SECONDARY = "#EF4444" # Crimson Red
COLOR_GREEN = "#10B981"     # Emerald Green
COLOR_YELLOW = "#F59E0B"    # Amber Yellow
COLOR_DARK = "#1E293B"      # Slate Grey

def plot_churn_distribution(df, target_col='Renewal_Status'):
    """
    Plots a pie chart of Renewed vs. Churned customers.
    """
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ['Status', 'Count']
    counts['Status'] = counts['Status'].map({1: 'Renewed (Loyal) 🟢', 0: 'Churned (At Risk) 🔴'})
    
    fig = px.pie(
        counts, 
        values='Count', 
        names='Status',
        title="Subscription Renewal Distribution",
        color='Status',
        color_discrete_map={
            'Renewed (Loyal) 🟢': COLOR_GREEN,
            'Churned (At Risk) 🔴': COLOR_SECONDARY
        },
        hole=0.4
    )
    fig.update_traces(textinfo='percent+value', pull=[0.02, 0.02])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family="Inter, sans-serif"),
        title_x=0.5
    )
    return fig

def plot_behavioral_analysis(df, numeric_col='Login_Frequency', target_col='Renewal_Status'):
    """
    Plots the relationship between a numeric feature (like Login Frequency)
    and the Renewal Status using box plots and histograms.
    """
    df_temp = df.copy()
    df_temp['Renewal_Status_Label'] = df_temp[target_col].map({1: 'Renewed', 0: 'Churned'})
    
    fig = px.box(
        df_temp,
        x='Renewal_Status_Label',
        y=numeric_col,
        color='Renewal_Status_Label',
        color_discrete_map={'Renewed': COLOR_GREEN, 'Churned': COLOR_SECONDARY},
        points="outliers",
        title=f"{numeric_col.replace('_', ' ')} Distribution by Renewal Status",
        labels={'Renewal_Status_Label': 'Renewal Status', numeric_col: numeric_col.replace('_', ' ')}
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_session_duration_distribution(df):
    """
    Compares average session duration between renewed and churned users using a histogram/density curve.
    """
    df_temp = df.copy()
    df_temp['Renewal_Status_Label'] = df_temp['Renewal_Status'].map({1: 'Renewed', 0: 'Churned'})
    
    fig = px.histogram(
        df_temp,
        x='Session_Duration',
        color='Renewal_Status_Label',
        barmode='overlay',
        color_discrete_map={'Renewed': COLOR_GREEN, 'Churned': COLOR_SECONDARY},
        marginal='box',
        title="Average Session Duration (Renewed vs. Churned)",
        labels={'Session_Duration': 'Session Duration (Minutes)', 'Renewal_Status_Label': 'Renewal Status'}
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_subscription_plan_analysis(df):
    """
    Shows renewal rates split by Subscription Plan (Basic vs Premium).
    """
    # Calculate renewal rate per subscription plan
    plan_rates = df.groupby('Subscription_Type')['Renewal_Status'].agg(['count', 'mean']).reset_index()
    plan_rates.columns = ['Subscription_Type', 'Total_Customers', 'Renewal_Rate']
    plan_rates['Renewal_Rate'] = plan_rates['Renewal_Rate'] * 100.0
    
    fig = px.bar(
        plan_rates,
        x='Subscription_Type',
        y='Renewal_Rate',
        text=plan_rates['Renewal_Rate'].apply(lambda x: f"{x:.1f}%"),
        title="Renewal Rates by Subscription Plan",
        color='Subscription_Type',
        color_discrete_map={'Basic': COLOR_PRIMARY, 'Premium': COLOR_DARK},
        labels={'Subscription_Type': 'Plan Type', 'Renewal_Rate': 'Renewal Rate (%)'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_range=[0, 110],
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    return fig

def plot_tenure_vs_probability(scored_df):
    """
    Plots customer age or tenure vs renewal probability.
    """
    # Bin tenure to make it smoother
    bins = np.arange(0, scored_df['Tenure'].max() + 3, 3)
    scored_df_binned = scored_df.copy()
    scored_df_binned['Tenure_Bin'] = pd.cut(scored_df_binned['Tenure'], bins=bins)
    
    grouped = scored_df_binned.groupby('Tenure_Bin', observed=False).agg({
        'Renewal_Probability': 'mean',
        'Customer_ID': 'count'
    }).reset_index()
    
    # Calculate midpoints of bins for scatter/line chart
    grouped['Tenure_Midpoint'] = grouped['Tenure_Bin'].apply(lambda x: x.mid if pd.notnull(x) else np.nan)
    grouped = grouped.dropna(subset=['Tenure_Midpoint'])
    
    fig = go.Figure()
    # Line chart of average probability
    fig.add_trace(go.Scatter(
        x=grouped['Tenure_Midpoint'],
        y=grouped['Renewal_Probability'] * 100.0,
        mode='lines+markers',
        name='Avg Probability',
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Renewal Probability by Tenure Cohort",
        xaxis_title="Tenure (Months)",
        yaxis_title="Average Renewal Probability (%)",
        yaxis_range=[0, 105],
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_confusion_matrix(cm):
    """
    Plots a confusion matrix heatmap.
    """
    x = ['Predicted Churn (0)', 'Predicted Renewal (1)']
    y = ['Actual Churn (0)', 'Actual Renewal (1)']
    
    # Format counts and percentages
    total = np.sum(cm)
    text = [
        [f"True Neg (TN)<br>{cm[0,0]} ({cm[0,0]/total:.1%})", f"False Pos (FP)<br>{cm[0,1]} ({cm[0,1]/total:.1%})"],
        [f"False Neg (FN)<br>{cm[1,0]} ({cm[1,0]/total:.1%})", f"True Pos (TP)<br>{cm[1,1]} ({cm[1,1]/total:.1%})"]
    ]
    
    fig = ff_create_annotated_heatmap(cm, x=x, y=y, annotation_text=text, colorscale='Blues')
    fig.update_layout(
        title="Confusion Matrix",
        font=dict(family="Inter, sans-serif")
    )
    return fig

def ff_create_annotated_heatmap(z, x, y, annotation_text, colorscale):
    """
    Helper function to draw an annotated heatmap manually using go.Heatmap
    to avoid extra dependencies on plotly.figure_factory.
    """
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=colorscale,
        showscale=False
    ))
    
    # Add annotations
    for i in range(len(y)):
        for j in range(len(x)):
            fig.add_annotation(
                x=x[j], y=y[i],
                text=annotation_text[i][j],
                showarrow=False,
                font=dict(color="black" if z[i][j] < (np.max(z)/1.5) else "white")
            )
            
    fig.update_layout(
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=400,
        margin=dict(l=100, r=100, t=50, b=50)
    )
    return fig

def plot_roc_curve(fpr, tpr, auc_score):
    """
    Plots the ROC curve for model evaluation.
    """
    fig = go.Figure()
    
    # Diagonal baseline (no predictive power)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Guess (AUC = 0.50)'
    ))
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color=COLOR_PRIMARY, width=3),
        name=f'Logistic Regression (AUC = {auc_score:.3f})'
    ))
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        xaxis_range=[-0.01, 1.01],
        yaxis_range=[-0.01, 1.01],
        font=dict(family="Inter, sans-serif"),
        legend=dict(x=0.55, y=0.1, bgcolor='rgba(255,255,255,0.7)')
    )
    return fig

def plot_feature_importance(coef_df):
    """
    Plots a horizontal bar chart of coefficients.
    """
    coef_df = coef_df.sort_values(by='Coefficient', ascending=True)
    
    # Color logic: green for renewal drivers (positive), red for risk drivers (negative)
    colors = [COLOR_GREEN if c >= 0 else COLOR_SECONDARY for c in coef_df['Coefficient']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=coef_df['Coefficient'],
        y=coef_df['Feature'].apply(lambda x: x.replace('_', ' ')),
        orientation='h',
        marker=dict(color=colors),
        text=coef_df['Coefficient'].apply(lambda x: f"{x:+.3f}"),
        textposition='outside'
    ))
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Feature Coefficients (Model Feature Importance)",
        xaxis_title="Logistic Regression Coefficient (Impact)",
        yaxis_title="Feature Name",
        font=dict(family="Inter, sans-serif"),
        height=min(600, 100 + len(coef_df) * 45),
        margin=dict(r=80) # Add margin for labels
    )
    return fig

def plot_risk_distribution(scored_df):
    """
    Plots the count of customers in each risk tier.
    """
    counts = scored_df['Risk_Level'].value_counts().reset_index()
    counts.columns = ['Risk_Level', 'Count']
    
    # Force order
    order = ['Likely Renewal 🟢', 'Medium Risk 🟡', 'High Churn Risk 🔴']
    counts['Risk_Level'] = pd.Categorical(counts['Risk_Level'], categories=order, ordered=True)
    counts = counts.sort_values('Risk_Level')
    
    fig = px.bar(
        counts,
        x='Risk_Level',
        y='Count',
        color='Risk_Level',
        text='Count',
        title="Customer Risk Segmentation Breakdown",
        color_discrete_map={
            'Likely Renewal 🟢': COLOR_GREEN,
            'Medium Risk 🟡': COLOR_YELLOW,
            'High Churn Risk 🔴': COLOR_SECONDARY
        },
        labels={'Risk_Level': 'Risk Tier', 'Count': 'Number of Customers'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_range=[0, counts['Count'].max() * 1.15],
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    return fig

def plot_forecast_trends(trend_df, metric='Renewal_Rate'):
    """
    Plots the historical vs forecasted monthly renewal rates or revenue.
    """
    fig = go.Figure()
    
    # Historical component
    hist = trend_df[trend_df['Type'] == 'Historical']
    fig.add_trace(go.Scatter(
        x=hist['Month_Label'],
        y=hist[metric] if metric == 'Revenue' else hist[metric] * 100.0,
        mode='lines+markers',
        name='Historical',
        line=dict(color=COLOR_DARK, width=3),
        marker=dict(size=8)
    ))
    
    # Forecasted component (including the connection point)
    fore = trend_df[trend_df['Type'] == 'Forecasted']
    # Insert last historical point to connect lines
    connection = hist.iloc[[-1]]
    fore_connected = pd.concat([connection, fore], ignore_index=True)
    
    fig.add_trace(go.Scatter(
        x=fore_connected['Month_Label'],
        y=fore_connected[metric] if metric == 'Revenue' else fore_connected[metric] * 100.0,
        mode='lines+markers',
        name='Linear Forecast',
        line=dict(color=COLOR_PRIMARY, width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    title = "Monthly Revenue Forecast (Linear Trend)" if metric == 'Revenue' else "Monthly Renewal Rate Forecast (%)"
    yaxis_title = "Monthly Revenue ($)" if metric == 'Revenue' else "Renewal Rate (%)"
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        xaxis_title="Time Period",
        yaxis_title=yaxis_title,
        font=dict(family="Inter, sans-serif"),
        legend=dict(x=0.02, y=0.95, bgcolor='rgba(255,255,255,0.7)')
    )
    
    return fig
