import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from PIL import Image

# Set page title
st.set_page_config(
    page_title="Sleep Health Analysis Dashboard",
    page_icon="💤",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    try:
        # Try to load processed data first
        data_path = os.path.join('data', 'processed', 'processed_sleep_data.csv')
        data = pd.read_csv(data_path)
    except:
        # If processed data is not available, load raw data
        data_path = os.path.join('data', 'raw', 'Sleep_health_and_lifestyle_dataset.csv')
        data = pd.read_csv(data_path)
    return data

sleep_data = load_data()

# Main title
st.title("💤 Sleep Health and Lifestyle Analysis Dashboard")
st.markdown("This dashboard provides insights from the Sleep Health and Lifestyle Dataset analysis.")

# Sidebar for filtering
st.sidebar.header("Filters")

# Gender filter
if 'Gender' in sleep_data.columns:
    gender_options = ['All'] + sorted(sleep_data['Gender'].unique().tolist())
    selected_gender = st.sidebar.selectbox('Gender', gender_options)

# Age range filter
if 'Age' in sleep_data.columns:
    min_age = int(sleep_data['Age'].min())
    max_age = int(sleep_data['Age'].max())
    age_range = st.sidebar.slider('Age Range', min_age, max_age, (min_age, max_age))

# BMI filter
if 'BMI Category' in sleep_data.columns:
    bmi_options = ['All'] + sorted(sleep_data['BMI Category'].unique().tolist())
    selected_bmi = st.sidebar.selectbox('BMI Category', bmi_options)

# Apply filters
filtered_data = sleep_data.copy()
if 'Gender' in sleep_data.columns and selected_gender != 'All':
    filtered_data = filtered_data[filtered_data['Gender'] == selected_gender]
if 'Age' in sleep_data.columns:
    filtered_data = filtered_data[(filtered_data['Age'] >= age_range[0]) & (filtered_data['Age'] <= age_range[1])]
if 'BMI Category' in sleep_data.columns and selected_bmi != 'All':
    filtered_data = filtered_data[filtered_data['BMI Category'] == selected_bmi]

# Display basic statistics
st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", len(filtered_data))

with col2:
    if 'Sleep Duration' in filtered_data.columns:
        avg_sleep = filtered_data['Sleep Duration'].mean()
        st.metric("Average Sleep (hrs)", f"{avg_sleep:.2f}")

with col3:
    if 'Quality of Sleep' in filtered_data.columns:
        avg_quality = filtered_data['Quality of Sleep'].mean()
        st.metric("Average Sleep Quality", f"{avg_quality:.2f}/10")

with col4:
    if 'Sleep Disorder' in filtered_data.columns:
        disorder_pct = (filtered_data['Sleep Disorder'] != 'None').mean() * 100
        st.metric("Sleep Disorder %", f"{disorder_pct:.1f}%")

# Show sleep quality distribution
st.header("Sleep Quality Analysis")
col1, col2 = st.columns(2)

with col1:
    if 'Quality of Sleep' in filtered_data.columns:
        st.subheader("Sleep Quality Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_data['Quality of Sleep'], kde=True, ax=ax)
        ax.set_xlabel('Quality of Sleep (1-10)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with col2:
    if 'Sleep Duration' in filtered_data.columns and 'Quality of Sleep' in filtered_data.columns:
        st.subheader("Sleep Duration vs Quality")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Sleep Duration', y='Quality of Sleep', 
                    hue='Gender' if 'Gender' in filtered_data.columns else None,
                    data=filtered_data, ax=ax)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Factors affecting sleep
st.header("Factors Affecting Sleep Quality")

# Sleep quality by occupation
if 'Occupation' in filtered_data.columns and 'Quality of Sleep' in filtered_data.columns:
    st.subheader("Sleep Quality by Occupation")
    
    occupation_data = filtered_data.groupby('Occupation')['Quality of Sleep'].mean().sort_values(ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Quality of Sleep', y='Occupation', data=occupation_data, ax=ax)
    ax.set_xlabel('Average Sleep Quality')
    ax.set_ylabel('Occupation')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Sleep quality by physical activity and stress
col1, col2 = st.columns(2)

with col1:
    if 'Physical Activity Level' in filtered_data.columns and 'Quality of Sleep' in filtered_data.columns:
        st.subheader("Impact of Physical Activity")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Physical Activity Level', y='Quality of Sleep', data=filtered_data, ax=ax)
        ax.set_xlabel('Physical Activity Level')
        ax.set_ylabel('Sleep Quality')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        x = filtered_data['Physical Activity Level']
        y = filtered_data['Quality of Sleep']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.7)
        
        correlation = filtered_data['Physical Activity Level'].corr(filtered_data['Quality of Sleep'])
        ax.set_title(f"Correlation: {correlation:.2f}")
        st.pyplot(fig)

with col2:
    if 'Stress Level' in filtered_data.columns and 'Quality of Sleep' in filtered_data.columns:
        st.subheader("Impact of Stress")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Stress Level', y='Quality of Sleep', data=filtered_data, ax=ax)
        ax.set_xlabel('Stress Level')
        ax.set_ylabel('Sleep Quality')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        x = filtered_data['Stress Level']
        y = filtered_data['Quality of Sleep']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.7)
        
        correlation = filtered_data['Stress Level'].corr(filtered_data['Quality of Sleep'])
        ax.set_title(f"Correlation: {correlation:.2f}")
        st.pyplot(fig)

# Sleep Disorders Analysis
if 'Sleep Disorder' in filtered_data.columns:
    st.header("Sleep Disorder Analysis")
    
    # Count of each sleep disorder
    disorder_counts = filtered_data['Sleep Disorder'].value_counts().reset_index()
    disorder_counts.columns = ['Sleep Disorder', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Sleep Disorders")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Sleep Disorder', data=disorder_counts, ax=ax)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        if 'Quality of Sleep' in filtered_data.columns:
            st.subheader("Sleep Quality by Disorder Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Sleep Disorder', y='Quality of Sleep', data=filtered_data, ax=ax)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# Personal Sleep Quality Predictor
st.header("Personal Sleep Quality Predictor")
st.markdown("Use the sliders below to predict your sleep quality based on your lifestyle factors.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 30)
    sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.1)
    physical_activity = st.slider("Physical Activity Level (1-100)", 10, 100, 50)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

with col2:
    heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 70)
    daily_steps = st.slider("Daily Steps", 1000, 20000, 8000, 500)
    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    gender = st.selectbox("Gender", ["Male", "Female"])

predict_button = st.button("Predict Sleep Quality")

if predict_button:
    # This is a simplified prediction for demonstration
    # In a real app, you'd load your trained model
    
    # Create a simple prediction based on factors
    base_score = 7.0
    
    # Adjust for sleep duration (optimal is 7-9 hours)
    if 7 <= sleep_duration <= 9:
        duration_effect = 0.5
    else:
        duration_effect = -0.5 * abs(sleep_duration - 8) / 2
    
    # Adjust for physical activity (higher is better)
    activity_effect = (physical_activity - 50) / 100
    
    # Adjust for stress (higher is worse)
    stress_effect = -(stress_level - 5) / 5
    
    # Adjust for daily steps (optimal around 10000)
    steps_effect = -abs(daily_steps - 10000) / 20000
    
    # Calculate predicted score
    predicted_score = base_score + duration_effect + activity_effect + stress_effect + steps_effect
    predicted_score = max(1, min(10, predicted_score))  # Ensure score is between 1-10
    
    st.success(f"Predicted Sleep Quality: {predicted_score:.1f}/10")
    
    # Show recommendations based on inputs
    st.subheader("Personalized Recommendations")
    recommendations = []
    
    if sleep_duration < 7:
        recommendations.append("Increase your sleep duration to at least 7 hours for better rest")
    elif sleep_duration > 9:
        recommendations.append("Consider slightly reducing your sleep duration to 7-9 hours for optimal rest")
    
    if physical_activity < 60:
        recommendations.append("Increase your physical activity level to improve sleep quality")
    
    if stress_level > 6:
        recommendations.append("Consider stress management techniques like meditation or deep breathing")
    
    if daily_steps < 7500:
        recommendations.append("Try to increase your daily steps to at least 7,500-10,000 for better sleep")
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    if not recommendations:
        st.write("Your sleep habits appear to be good! Maintain your current routine for quality rest.")

# Footer
st.markdown("---")
st.markdown("Sleep Health Analysis Dashboard | Created as part of the Sleep Quality Analysis Project")