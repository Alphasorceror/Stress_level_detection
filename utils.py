"""
Utility functions for the stress level detection dashboard.
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler

def load_data(file_path="stress.csv", use_db=True):
    """
    Load the stress dataset from either the database or a CSV file.
    
    Args:
        file_path (str): Path to the CSV file (used if use_db is False)
        use_db (bool): Whether to use the database to load data
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if use_db:
        try:
            # Import the database module only when needed
            from database import check_data_exists, load_data_from_db, load_csv_to_db
            
            # Check if data exists in the database
            if check_data_exists():
                # Load data from the database
                df = load_data_from_db()
                print("Data loaded from database successfully.")
                return df
            else:
                # If not, load from CSV and then save to database
                try:
                    df = pd.read_csv(file_path)
                    # Save to database
                    if load_csv_to_db(file_path):
                        print(f"Data loaded from {file_path} and saved to database.")
                    else:
                        print(f"Data loaded from {file_path} but failed to save to database.")
                    return df
                except Exception as e:
                    print(f"Error loading CSV data: {e}")
                    return None
        except Exception as e:
            print(f"Error using database: {e}")
            # Fall back to CSV if there's a database error
            try:
                df = pd.read_csv(file_path)
                print(f"Fallback: Data loaded from {file_path}.")
                return df
            except Exception as e2:
                print(f"Error loading data: {e2}")
                return None
    else:
        # Load directly from CSV
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded from {file_path}.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

def get_feature_categories():
    """
    Return a dictionary of feature categories and their corresponding columns.
    
    Returns:
        dict: Dictionary of feature categories and their columns
    """
    feature_categories = {
        'Social': ['social_timestamp', 'social_karma', 'social_upvote_ratio', 'social_num_comments'],
        'Syntax': ['syntax_ari', 'syntax_fk_grade'],
        'Sentiment': ['sentiment', 'lex_liwc_Tone'],
        'Linguistic': [
            'lex_liwc_WC', 'lex_liwc_Analytic', 'lex_liwc_Clout', 
            'lex_liwc_Authentic', 'lex_liwc_WPS', 'lex_liwc_Sixltr', 'lex_liwc_Dic'
        ],
        'Pronouns': [
            'lex_liwc_pronoun', 'lex_liwc_ppron', 'lex_liwc_i', 'lex_liwc_we', 
            'lex_liwc_you', 'lex_liwc_shehe', 'lex_liwc_they', 'lex_liwc_ipron'
        ],
        'Parts of Speech': [
            'lex_liwc_article', 'lex_liwc_prep', 'lex_liwc_auxverb', 
            'lex_liwc_adverb', 'lex_liwc_conj', 'lex_liwc_negate', 'lex_liwc_verb',
            'lex_liwc_adj', 'lex_liwc_compare', 'lex_liwc_interrog', 'lex_liwc_number',
            'lex_liwc_quant'
        ],
        'Psychological': [
            'lex_liwc_affect', 'lex_liwc_posemo', 'lex_liwc_negemo', 'lex_liwc_anx', 
            'lex_liwc_anger', 'lex_liwc_sad', 'lex_liwc_social', 'lex_liwc_family', 
            'lex_liwc_friend', 'lex_liwc_female', 'lex_liwc_male', 'lex_liwc_cogproc',
            'lex_liwc_insight', 'lex_liwc_cause', 'lex_liwc_discrep', 'lex_liwc_tentat',
            'lex_liwc_certain', 'lex_liwc_differ', 'lex_liwc_percept', 'lex_liwc_see',
            'lex_liwc_hear', 'lex_liwc_feel', 'lex_liwc_bio', 'lex_liwc_body',
            'lex_liwc_health', 'lex_liwc_sexual', 'lex_liwc_ingest', 'lex_liwc_drives',
            'lex_liwc_affiliation', 'lex_liwc_achieve', 'lex_liwc_power', 'lex_liwc_reward',
            'lex_liwc_risk'
        ],
        'Time and Space': [
            'lex_liwc_focuspast', 'lex_liwc_focuspresent', 'lex_liwc_focusfuture',
            'lex_liwc_relativ', 'lex_liwc_motion', 'lex_liwc_space', 'lex_liwc_time'
        ],
        'Personal Concerns': [
            'lex_liwc_work', 'lex_liwc_leisure', 'lex_liwc_home', 'lex_liwc_money',
            'lex_liwc_relig', 'lex_liwc_death'
        ],
        'Informal Language': [
            'lex_liwc_informal', 'lex_liwc_swear', 'lex_liwc_netspeak',
            'lex_liwc_assent', 'lex_liwc_nonflu', 'lex_liwc_filler'
        ],
        'Dictionary of Affect': [
            'lex_dal_pleasantness', 'lex_dal_activation', 'lex_dal_imagery',
            'lex_dal_max_pleasantness', 'lex_dal_max_activation', 'lex_dal_max_imagery',
            'lex_dal_min_pleasantness', 'lex_dal_min_activation', 'lex_dal_min_imagery',
            'lex_dal_avg_activation', 'lex_dal_avg_imagery', 'lex_dal_avg_pleasantness'
        ]
    }
    return feature_categories

def clean_text(text):
    """
    Clean the text by removing URLs, special characters, etc.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text
    return ""

def convert_timestamp_to_date(df, timestamp_col='social_timestamp'):
    """
    Convert Unix timestamp to datetime.
    
    Args:
        df (pd.DataFrame): DataFrame containing timestamp column
        timestamp_col (str): Name of the timestamp column
        
    Returns:
        pd.DataFrame: DataFrame with added date column
    """
    df_copy = df.copy()
    if timestamp_col in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy[timestamp_col], unit='s')
    return df_copy

def get_stress_indicators(df):
    """
    Extract key stress indicators from the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with stress data
        
    Returns:
        dict: Dictionary of stress indicators
    """
    indicators = {}
    
    if 'label' in df.columns:
        indicators['Stress Prevalence'] = df['label'].mean() * 100
    
    if 'lex_liwc_anx' in df.columns:
        indicators['Anxiety Score (Avg)'] = df['lex_liwc_anx'].mean()
    
    if 'lex_liwc_negemo' in df.columns:
        indicators['Negative Emotion (Avg)'] = df['lex_liwc_negemo'].mean()
    
    if 'sentiment' in df.columns:
        indicators['Average Sentiment'] = df['sentiment'].mean()
    
    return indicators

def get_subreddit_stats(df):
    """
    Get stress statistics by subreddit.
    
    Args:
        df (pd.DataFrame): DataFrame with stress data
        
    Returns:
        pd.DataFrame: Statistics by subreddit
    """
    if 'subreddit' in df.columns and 'label' in df.columns:
        # Group by subreddit and calculate statistics
        stats = df.groupby('subreddit').agg({
            'label': ['mean', 'count'],
            'lex_liwc_anx': 'mean',
            'lex_liwc_negemo': 'mean',
            'sentiment': 'mean'
        }).reset_index()
        
        # Flatten multi-level column names
        stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stats.columns.values]
        
        # Rename columns for clarity
        stats = stats.rename(columns={
            'label_mean': 'stress_rate',
            'label_count': 'post_count',
            'lex_liwc_anx_mean': 'avg_anxiety',
            'lex_liwc_negemo_mean': 'avg_negative_emotion',
            'sentiment_mean': 'avg_sentiment'
        })
        
        # Calculate stress rate as percentage
        stats['stress_rate'] = stats['stress_rate'] * 100
        
        return stats
    
    return pd.DataFrame()

def scale_features(df, features):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df (pd.DataFrame): DataFrame with features to scale
        features (list): List of feature names to scale
        
    Returns:
        pd.DataFrame: DataFrame with scaled features
        StandardScaler: Fitted scaler
    """
    # Check if all features exist in the DataFrame
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        return df, None
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_scaled = df.copy()
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    df_scaled[valid_features] = scaler.fit_transform(df[valid_features])
    
    return df_scaled, scaler