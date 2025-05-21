# src/feature_utils.py
import pandas as pd
import numpy as np
from scipy import stats


def calculate_sleep_score(df, quality_col='Quality of Sleep', duration_col='Sleep Duration', 
                          efficiency_col=None, deep_sleep_col=None):
    """
    Calculate a composite sleep score based on available metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    quality_col : str
        Column name for sleep quality
    duration_col : str
        Column name for sleep duration
    efficiency_col : str or None
        Column name for sleep efficiency
    deep_sleep_col : str or None
        Column name for deep sleep percentage, if available
        
    Returns:
    --------
    pandas.Series
        Sleep score for each record
    """
    # Create a copy to avoid modifying the original
    df_score = df.copy()
    
    # Initialize weights and components
    weights = {}
    score_components = {}
    
    # Handle sleep quality
    if quality_col in df_score.columns:
        weights[quality_col] = 0.5
        # Scale to 0-100 if needed
        if df_score[quality_col].max() <= 10:
            score_components[quality_col] = df_score[quality_col] * 10
        else:
            score_components[quality_col] = df_score[quality_col]
    
    # Handle sleep duration (assuming 7-9 hours is ideal)
    if duration_col in df_score.columns:
        weights[duration_col] = 0.5
        # Score calculation for duration: 100% for 7-9 hours, less for too short or too long
        score_components[duration_col] = df_score[duration_col].apply(
            lambda x: 100 if 7 <= x <= 9 else 100 - (20 * abs(x - 8))
        ).clip(0, 100)
    
    # Handle sleep efficiency
    if efficiency_col and efficiency_col in df_score.columns:
        # Adjust weights
        for key in weights:
            weights[key] = 0.33
        weights[efficiency_col] = 0.34
        
        # Good efficiency is above 85%
        score_components[efficiency_col] = (df_score[efficiency_col] * 100).clip(0, 100)
    
    # Handle deep sleep percentage
    if deep_sleep_col and deep_sleep_col in df_score.columns:
        # Adjust weights
        for key in weights:
            weights[key] = 0.25
        weights[deep_sleep_col] = 0.25
        
        # Ideal deep sleep is 20-25% of total sleep
        score_components[deep_sleep_col] = df_score[deep_sleep_col].apply(
            lambda x: 100 if 20 <= x <= 25 else 100 - (10 * abs(x - 22.5))
        ).clip(0, 100)
    
    # Calculate weighted score
    sleep_score = pd.Series(0, index=df_score.index)
    for col, weight in weights.items():
        if col in score_components:
            sleep_score += weight * score_components[col]
    
    return sleep_score


def identify_sleep_patterns(df, user_id_col='Person ID', date_col=None, sleep_metrics=None):
    """
    Identify sleep patterns over time or by individual.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    user_id_col : str or None
        Column name for user ID
    date_col : str or None
        Column name for date/time
    sleep_metrics : list or None
        List of sleep metric columns to analyze
        
    Returns:
    --------
    dict
        Dictionary of sleep patterns
    """
    patterns = {}
    
    if sleep_metrics is None:
        sleep_metrics = ['Sleep Duration', 'Quality of Sleep']
        if 'Sleep Efficiency' in df.columns:
            sleep_metrics.append('Sleep Efficiency')
    
    sleep_metrics = [col for col in sleep_metrics if col in df.columns]
    
    # User-specific patterns
    if user_id_col and user_id_col in df.columns:
        user_patterns = {}
        
        for metric in sleep_metrics:
            user_avg = df.groupby(user_id_col)[metric].mean()
            user_patterns[metric] = user_avg
        
        patterns['user'] = user_patterns
    
    # Weekly patterns (if date column exists)
    if date_col and date_col in df.columns:
        df['DayOfWeek'] = pd.to_datetime(df[date_col]).dt.day_name()
        
        # Average sleep metrics by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_patterns = {}
        
        for metric in sleep_metrics:
            weekly_avg = df.groupby('DayOfWeek')[metric].mean().reindex(day_order)
            weekly_patterns[metric] = weekly_avg
        
        patterns['weekly'] = weekly_patterns
    
    # Correlation between metrics
    correlation_matrix = df[sleep_metrics].corr()
    patterns['correlations'] = correlation_matrix
    
    return patterns


def analyze_factors_impact(df, sleep_quality_col='Quality of Sleep'):
    """
    Analyze the impact of various factors on sleep quality.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    sleep_quality_col : str
        Column name for sleep quality
        
    Returns:
    --------
    pandas.DataFrame
        Impact analysis results
    """
    # Identify relevant factors in the dataset
    potential_factors = [
        'Physical Activity Level', 'Stress Level', 'Heart Rate', 
        'Daily Steps', 'Age', 'Has Sleep Disorder', 'BP Risk', 
        'BMI Numeric', 'Stress-Activity Ratio', 'Cardio Health Score'
    ]
    
    # Filter for factors that exist in the dataset
    factors = [col for col in potential_factors if col in df.columns]
    
    if not factors:
        return pd.DataFrame(columns=['Factor', 'Correlation', 'P-Value'])
    
    results = []
    
    for factor in factors:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[factor]):
            continue
            
        # Calculate correlation
        correlation, p_value = stats.pearsonr(df[factor], df[sleep_quality_col])
        
        results.append({
            'Factor': factor,
            'Correlation': correlation,
            'P-Value': p_value,
            'Significant': p_value < 0.05,
            'Impact': 'Positive' if correlation > 0 else 'Negative'
        })
    
    # Add categorical analyses
    categorical_factors = ['Gender', 'BMI Category', 'Occupation', 'Age Group', 'Sleep Disorder']
    for factor in [col for col in categorical_factors if col in df.columns]:
        # Calculate ANOVA F-test for categorical variables
        categories = df[factor].unique()
        if len(categories) >= 2:  # Need at least 2 categories
            groups = [df[df[factor] == cat][sleep_quality_col].values for cat in categories]
            f_stat, p_value = stats.f_oneway(*groups)
            
            results.append({
                'Factor': factor,
                'Correlation': None,  # Not applicable for categorical variables
                'F-Statistic': f_stat,
                'P-Value': p_value,
                'Significant': p_value < 0.05,
                'Impact': 'Varies by category' if p_value < 0.05 else 'Not significant'
            })
    
    return pd.DataFrame(results).sort_values('P-Value')