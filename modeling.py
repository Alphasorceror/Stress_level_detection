"""
Machine learning models for stress level detection.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

def train_model(X_train, y_train, model_type='random_forest', random_state=42):
    """
    Train a machine learning model for stress detection.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        model_type (str): Type of model to train ('random_forest', 'logistic', 'svm', 'gradient_boosting')
        random_state (int): Random state for reproducibility
        
    Returns:
        object: Trained model
    """
    if X_train is None or y_train is None:
        return None
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=5, 
            random_state=random_state
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=random_state
        )
    elif model_type == 'svm':
        model = SVC(
            C=1.0, 
            kernel='rbf', 
            probability=True, 
            random_state=random_state
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=random_state
        )
    else:
        # Default to Random Forest
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state
        )
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    if model is None or X_test is None or y_test is None:
        return {}
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary')
    }
    
    return metrics

def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        
    Returns:
        dict: Dictionary of feature importance scores
    """
    if model is None or not feature_names:
        return {}
    
    try:
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        # Check if model has coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return {}
        
        # Create dictionary of feature importance
        feature_importance = dict(zip(feature_names, importance))
        return feature_importance
    except:
        return {}

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix for the model predictions.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        
    Returns:
        bytes: Image bytes of the confusion matrix plot
    """
    if model is None or X_test is None or y_test is None:
        return None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Save figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Close the figure to free memory
    plt.close()
    
    return buf

def plot_feature_importance(feature_importance, top_n=10):
    """
    Plot feature importance.
    
    Args:
        feature_importance (dict): Dictionary of feature importance scores
        top_n (int): Number of top features to plot
        
    Returns:
        bytes: Image bytes of the feature importance plot
    """
    if not feature_importance:
        return None
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Take top N features
    top_features = sorted_features[:top_n]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Extract feature names and importance values
    feature_names = [item[0] for item in top_features]
    importance_values = [item[1] for item in top_features]
    
    # Create horizontal bar plot
    sns.barplot(x=importance_values, y=feature_names, palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    
    # Save figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Close the figure to free memory
    plt.close()
    
    return buf

def predict_stress(model, X):
    """
    Predict stress level for a single input.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        
    Returns:
        tuple: (Predicted label, probability)
    """
    if model is None or X is None:
        return None, None
    
    # Make prediction
    pred_label = model.predict(X)[0]
    
    # Get probability
    if hasattr(model, 'predict_proba'):
        pred_prob = model.predict_proba(X)[0][1]  # Probability of class 1
    else:
        pred_prob = None
    
    return pred_label, pred_prob

def cross_validate_model(X, y, model_type='random_forest', cv=5, random_state=42):
    """
    Perform cross-validation on the model.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        model_type (str): Type of model to train
        cv (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (Mean CV score, standard deviation)
    """
    if X is None or y is None:
        return 0, 0
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=random_state
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=random_state
        )
    elif model_type == 'svm':
        model = SVC(
            C=1.0, 
            kernel='rbf', 
            random_state=random_state
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=random_state
        )
    else:
        # Default to Random Forest
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state
        )
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    
    return cv_scores.mean(), cv_scores.std()