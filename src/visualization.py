# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_plot_style():
    """Set the default style for matplotlib plots."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_sleep_distribution(df, column, title=None, save_path=None):
    """
    Plot distribution of a sleep metric.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    column : str
        Column name for the sleep metric
    title : str or None
        Plot title
    save_path : str or None
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_plot_style()
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot histogram with KDE
    sns.histplot(df[column], kde=True, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.set_title(title or f'Distribution of {column}')
    
    # Add mean line
    mean_val = df[column].mean()
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7)
    ax.text(mean_val, ax.get_ylim()[1]*0.9, f' Mean: {mean_val:.2f}', 
            color='red', ha='left', va='top')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_correlation_heatmap(df, columns=None, title=None, save_path=None):
    """
    Create a correlation heatmap for sleep metrics and factors.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    columns : list or None
        List of column names to include in correlation
    title : str or None
        Plot title
    save_path : str or None
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_plot_style()
    
    # Use specified columns or all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    
    # Set title
    plt.title(title or 'Correlation Heatmap')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_boxplot_by_category(df, x_col, y_col, title=None, save_path=None):
    """
    Create a boxplot of a sleep metric by categorical variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    x_col : str
        Column name for the categorical variable
    y_col : str
        Column name for the sleep metric
    title : str or None
        Plot title
    save_path : str or None
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_plot_style()
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Create boxplot
    sns.boxplot(x=x_col, y=y_col, data=df, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f'{y_col} by {x_col}')
    
    # Rotate x-axis labels if there are many categories
    if df[x_col].nunique() > 4:
        plt.xticks(rotation=45, ha='right')
    
    # Add mean values as text
    means = df.groupby(x_col)[y_col].mean()
    for i, mean_val in enumerate(means):
        ax.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_scatter_with_trendline(df, x_col, y_col, hue_col=None, title=None, save_path=None):
    """
    Create a scatter plot with trendline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    hue_col : str or None
        Column name for color coding points
    title : str or None
        Plot title
    save_path : str or None
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_plot_style()
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Create scatter plot
    sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, alpha=0.7, ax=ax)
    
    # Add trendline if no hue or if hue has only one value
    if hue_col is None or df[hue_col].nunique() == 1:
        # Use numpy to fit a linear regression
        x = df[x_col]
        y = df[y_col]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Plot trendline
        ax.plot(x, p(x), "r--", alpha=0.7)
        
        # Display correlation
        correlation = df[x_col].corr(df[y_col])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                 transform=ax.transAxes, ha='left', va='top')
    
    # Set labels and title
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f'{y_col} vs {x_col}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cluster_comparison(df, cluster_col, feature_cols, title=None, save_path=None):
    """
    Create a radar chart or parallel coordinates plot to compare clusters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sleep data with cluster assignments
    cluster_col : str
        Column name for cluster assignments
    feature_cols : list
        List of feature column names to compare
    title : str or None
        Plot title
    save_path : str or None
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_plot_style()
    
    # Get cluster profiles
    cluster_profiles = df.groupby(cluster_col)[feature_cols].mean()
    
    # Normalize the features for better visualization
    scaler = lambda x: (x - x.min()) / (x.max() - x.min())
    cluster_profiles_norm = cluster_profiles.apply(scaler)
    
    # Create a bar chart for comparing clusters
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the cluster profiles
    cluster_profiles_norm.T.plot(kind='bar', ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Feature')
    ax.set_ylabel('Normalized Value')
    ax.set_title(title or f'Cluster Profiles Comparison')
    
    # Adjust legend and labels
    ax.legend(title=cluster_col)
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(feature_importance_df, title=None, save_path=None):
    """
    Plot feature importance from a machine learning model.
    
    Parameters:
    -----------
    feature_importance_df : pandas.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    title : str or None
        Plot title
    save_path : str or None
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_plot_style()
    
    # Sort by importance
    sorted_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Limit to top 15 features if there are many
    if len(sorted_df) > 15:
        sorted_df = sorted_df.head(15)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    sns.barplot(x='Importance', y='Feature', data=sorted_df, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title or 'Feature Importance')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig