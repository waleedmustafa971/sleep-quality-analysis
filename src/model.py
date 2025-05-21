# src/model.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


def identify_sleep_clusters(df, features, n_clusters=3):
    """
    Identify sleep behavior clusters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    features : list
        List of features to use for clustering
    n_clusters : int
        Number of clusters to identify
        
    Returns:
    --------
    tuple
        (KMeans model, DataFrame with cluster assignments, Cluster profiles)
    """
    # Select features for clustering
    feature_cols = [col for col in features if col in df.columns]
    X = df[feature_cols].copy()
    
    # Handle missing values if any
    X.fillna(X.mean(), inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Sleep Cluster'] = clusters
    
    # Generate cluster profiles
    cluster_profiles = df_with_clusters.groupby('Sleep Cluster')[feature_cols].mean()
    
    print(f"Identified {n_clusters} sleep behavior clusters")
    print("Cluster profiles:")
    print(cluster_profiles)
    
    return kmeans, df_with_clusters, cluster_profiles


def train_sleep_quality_model(df, target_col, feature_cols):
    """
    Train a model to predict sleep quality based on daily habits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    target_col : str
        Column name for sleep quality (target variable)
    feature_cols : list
        List of feature columns for prediction
        
    Returns:
    --------
    tuple
        (Trained model, Training metrics)
    """
    # Select only features that exist in the dataset
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Select features and target
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': available_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Metrics
    metrics = {
        'RMSE': rmse,
        'R^2': r2,
        'Feature Importance': feature_importance
    }
    
    print(f"Model trained to predict {target_col}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print("Top features by importance:")
    print(feature_importance.head())
    
    return model, metrics


def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    file_path : str
        Path to save the model
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def load_model(file_path):
    """
    Load a trained model from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the model file
        
    Returns:
    --------
    sklearn model
        Loaded model
    """
    try:
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def generate_sleep_recommendations(df, cluster_profiles, user_cluster):
    """
    Generate personalized sleep recommendations based on cluster and patterns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    cluster_profiles : pandas.DataFrame
        Profiles of sleep clusters
    user_cluster : int
        Cluster assignment for the user
        
    Returns:
    --------
    list
        List of personalized recommendations
    """
    recommendations = []
    
    # Get user's cluster profile
    user_profile = cluster_profiles.loc[user_cluster]
    
    # Get ideal cluster (assuming highest sleep quality is best)
    if 'Quality of Sleep' in cluster_profiles.columns:
        ideal_cluster = cluster_profiles['Quality of Sleep'].idxmax()
        ideal_profile = cluster_profiles.loc[ideal_cluster]
        
        # Sleep duration recommendations
        if 'Sleep Duration' in user_profile.index:
            if user_profile['Sleep Duration'] < 7:
                recommendations.append(f"Increase sleep duration to at least 7 hours (current avg: {user_profile['Sleep Duration']:.1f} hours)")
            elif user_profile['Sleep Duration'] > 9:
                recommendations.append(f"Consider reducing sleep duration to 7-9 hours (current avg: {user_profile['Sleep Duration']:.1f} hours)")
            else:
                recommendations.append(f"Maintain current sleep duration of {user_profile['Sleep Duration']:.1f} hours")
        
        # Physical activity recommendations
        if 'Physical Activity Level' in user_profile.index:
            if user_profile['Physical Activity Level'] < 60:
                recommendations.append(f"Increase physical activity level (current: {user_profile['Physical Activity Level']:.1f}/100)")
            else:
                recommendations.append(f"Maintain your good physical activity level of {user_profile['Physical Activity Level']:.1f}/100")
        
        # Stress management recommendations
        if 'Stress Level' in user_profile.index:
            if user_profile['Stress Level'] > 5:
                recommendations.append(f"Implement stress reduction techniques (current stress level: {user_profile['Stress Level']:.1f}/10)")
            else:
                recommendations.append(f"Continue your effective stress management (current level: {user_profile['Stress Level']:.1f}/10)")
        
        # Daily steps recommendations
        if 'Daily Steps' in user_profile.index:
            if user_profile['Daily Steps'] < 8000:
                recommendations.append(f"Increase daily steps to at least 8,000-10,000 (current: {user_profile['Daily Steps']:.0f})")
            else:
                recommendations.append(f"Maintain your good activity level of {user_profile['Daily Steps']:.0f} daily steps")
        
        # Heart rate recommendations
        if 'Heart Rate' in user_profile.index:
            if user_profile['Heart Rate'] > 80:
                recommendations.append(f"Consider cardiovascular activities to lower resting heart rate (current: {user_profile['Heart Rate']:.0f} bpm)")
    
    # Add general recommendations if the list is too short
    if len(recommendations) < 3:
        general_recommendations = [
            "Maintain a consistent sleep schedule, even on weekends",
            "Keep your bedroom cool, dark, and quiet",
            "Avoid large meals, caffeine, and alcohol before bedtime",
            "Create a relaxing pre-sleep routine",
            "Ensure your mattress and pillows are comfortable and supportive",
            "Limit screen time 1-2 hours before bed",
            "Practice relaxation techniques like deep breathing or meditation before sleep",
            "Exercise regularly, but avoid intense workouts close to bedtime",
            "Keep a sleep journal to track patterns and improvements"
        ]
        
        needed_recs = 3 - len(recommendations)
        recommendations.extend(general_recommendations[:needed_recs])
    
    return recommendations


def predict_sleep_quality(model, user_data):
    """
    Predict sleep quality for a given user data.
    
    Parameters:
    -----------
    model : sklearn model
        Trained predictive model
    user_data : dict or pandas.DataFrame
        User data containing feature values
        
    Returns:
    --------
    float
        Predicted sleep quality
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(user_data, dict):
            user_df = pd.DataFrame([user_data])
        else:
            user_df = user_data.copy()
        
        # Get the model's feature names
        feature_names = model.feature_names_in_
        
        # Check if all required features are present
        missing_features = [feat for feat in feature_names if feat not in user_df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with mean values or zeros
            for feat in missing_features:
                user_df[feat] = 0
        
        # Select only the features used by the model
        user_features = user_df[feature_names].copy()
        
        # Make prediction
        prediction = model.predict(user_features)[0]
        
        return prediction
    
    except Exception as e:
        print(f"Error predicting sleep quality: {e}")
        # Return a reasonable fallback value
        return 5.0  # Middle of the typical 1-10 scale


def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate the performance of a trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained predictive model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Calculate explained variance
    explained_var = 1 - np.var(y_test - y_pred) / np.var(y_test)
    
    # Result dictionary
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2,
        'Explained Variance': explained_var
    }
    
    return metrics


def analyze_sleep_disorders(df, sleep_quality_col='Quality of Sleep'):
    """
    Analyze the impact of sleep disorders on sleep quality.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    sleep_quality_col : str
        Column name for sleep quality
        
    Returns:
    --------
    dict
        Dictionary of analysis results
    """
    results = {}
    
    # Check if Sleep Disorder column exists
    if 'Sleep Disorder' not in df.columns:
        return {'error': 'Sleep Disorder column not found in dataset'}
    
    # Group by sleep disorder and calculate statistics
    disorder_stats = df.groupby('Sleep Disorder')[sleep_quality_col].agg(['mean', 'std', 'count']).reset_index()
    results['stats'] = disorder_stats
    
    # Calculate percentage of people with each disorder
    total_count = len(df)
    disorder_counts = df['Sleep Disorder'].value_counts().reset_index()
    disorder_counts.columns = ['Sleep Disorder', 'Count']
    disorder_counts['Percentage'] = (disorder_counts['Count'] / total_count) * 100
    results['counts'] = disorder_counts
    
    # Analyze characteristics of people with disorders vs. without
    has_disorder = df['Sleep Disorder'] != 'None'
    disorder_comparison = df.groupby(has_disorder).mean()
    disorder_comparison.index = ['No Disorder', 'Has Disorder']
    results['comparison'] = disorder_comparison
    
    return results