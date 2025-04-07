"""
Data preprocessing functions for the stress level detection dashboard.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import clean_text, get_feature_categories

def prepare_data(df, text_column='text', label_column='label'):
    """
    Prepare the data for analysis and modeling.
    
    Args:
        df (pd.DataFrame): Raw stress dataset
        text_column (str): Name of the text column
        label_column (str): Name of the label column
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    if df is None or df.empty:
        return None
    
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Clean text data
    if text_column in data.columns:
        data['clean_text'] = data[text_column].apply(clean_text)
    
    # Convert timestamp to datetime
    if 'social_timestamp' in data.columns:
        data['date'] = pd.to_datetime(data['social_timestamp'], unit='s')
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['day_of_week'] = data['date'].dt.dayofweek
    
    # Handle missing values in numerical columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if not numerical_cols.empty:
        imputer = SimpleImputer(strategy='median')
        data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    
    # Encode categorical columns if needed
    categorical_cols = data.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != text_column and col != 'date']
    
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    
    return data

def select_features(df, label_column='label', k=20, exclude_cols=None):
    """
    Select the most relevant features using SelectKBest.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        label_column (str): Name of the label column
        k (int): Number of features to select
        exclude_cols (list): List of columns to exclude from feature selection
        
    Returns:
        tuple: (list of selected features, feature importance scores)
    """
    if df is None or df.empty or label_column not in df.columns:
        return [], {}
    
    if exclude_cols is None:
        exclude_cols = ['text', 'clean_text', 'date', 'id', 'post_id', 'subreddit', 'sentence_range']
    
    # Identify numerical features
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in feature_cols if col != label_column and col not in exclude_cols]
    
    if len(feature_cols) == 0:
        return [], {}
    
    # Prepare X and y
    X = df[feature_cols]
    y = df[label_column]
    
    # Apply feature selection
    selector = SelectKBest(f_classif, k=min(k, len(feature_cols)))
    selector.fit(X, y)
    
    # Get selected features and their scores
    feature_scores = dict(zip(feature_cols, selector.scores_))
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    
    return selected_features, feature_scores

def split_dataset(df, features, label_column='label', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        features (list): List of feature columns
        label_column (str): Name of the label column
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if df is None or df.empty or label_column not in df.columns:
        return None, None, None, None
    
    # Ensure all features exist in the dataframe
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        return None, None, None, None
    
    # Prepare X and y
    X = df[valid_features]
    y = df[label_column]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def get_feature_importance_df(feature_scores):
    """
    Create a dataframe of feature importance scores.
    
    Args:
        feature_scores (dict): Dictionary of feature scores
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if not feature_scores:
        return pd.DataFrame()
    
    # Create a dataframe from feature scores
    importance_df = pd.DataFrame({
        'Feature': list(feature_scores.keys()),
        'Importance': list(feature_scores.values())
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Add feature categories
    feature_categories = get_feature_categories()
    
    # Create a mapping from feature to category
    feature_to_category = {}
    for category, features in feature_categories.items():
        for feature in features:
            feature_to_category[feature] = category
    
    # Add category column
    importance_df['Category'] = importance_df['Feature'].map(
        lambda x: feature_to_category.get(x, 'Other')
    )
    
    return importance_df