def create_sleep_features(df):
    """
    Create additional features for sleep analysis specific to the Sleep Health and Lifestyle dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned sleep data
        
    Returns:
    --------
    pandas.DataFrame
        Sleep data with additional features
    """
    df_features = df.copy()
    
    # Sleep efficiency score - combining duration and quality
    if 'Sleep Duration' in df_features.columns and 'Quality of Sleep' in df_features.columns:
        # Normalize sleep duration (7-9 hours is ideal)
        df_features['Sleep Duration Score'] = df_features['Sleep Duration'].apply(
            lambda x: 10 if 7 <= x <= 9 else 10 - 2 * abs(x - 8)
        ).clip(0, 10)
        
        # Combined sleep score (50% quality, 50% duration)
        df_features['Sleep Score'] = (df_features['Quality of Sleep'] + df_features['Sleep Duration Score']) / 2
    
    # Stress-activity balance (physical activity relative to stress level)
    if 'Physical Activity Level' in df_features.columns and 'Stress Level' in df_features.columns:
        df_features['Stress-Activity Ratio'] = df_features['Stress Level'] / df_features['Physical Activity Level']
    
    # Health metrics composite
    if 'BMI Category' in df_features.columns:
        # Convert BMI category to numeric
        bmi_mapping = {
            'Normal': 0, 
            'Normal Weight': 0,
            'Overweight': 1, 
            'Obese': 2,
            'Underweight': 1
        }
        df_features['BMI Numeric'] = df_features['BMI Category'].map(bmi_mapping)
    
    # Blood pressure risk factor
    if 'Blood Pressure' in df_features.columns:
        # Extract systolic and diastolic
        df_features[['Systolic', 'Diastolic']] = df_features['Blood Pressure'].str.split('/', expand=True).astype(int)
        
        # Create blood pressure risk score
        df_features['BP Risk'] = ((df_features['Systolic'] > 130).astype(int) + 
                                 (df_features['Diastolic'] > 85).astype(int))
    
    # Cardiovascular health score
    if 'Heart Rate' in df_features.columns and 'Daily Steps' in df_features.columns:
        # Normalize heart rate (60-70 is ideal)
        df_features['Heart Rate Score'] = 10 - abs(df_features['Heart Rate'] - 65) / 5
        
        # Normalize daily steps (8000-12000 is ideal)
        df_features['Activity Score'] = df_features['Daily Steps'].apply(
            lambda x: 10 if 8000 <= x <= 12000 else 10 - abs(x - 10000) / 1000
        ).clip(0, 10)
        
        # Combined cardiovascular score
        df_features['Cardio Health Score'] = (df_features['Heart Rate Score'] + df_features['Activity Score']) / 2
    
    # Sleep disorder binary flag
    if 'Sleep Disorder' in df_features.columns:
        df_features['Has Sleep Disorder'] = (df_features['Sleep Disorder'] != 'None').astype(int)
    
    # Age groups
    if 'Age' in df_features.columns:
        df_features['Age Group'] = pd.cut(
            df_features['Age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=['<30', '30-40', '40-50', '50-60', '60+']
        )
    
    print(f"Created {len(df_features.columns) - len(df.columns)} new features")
    return df_features