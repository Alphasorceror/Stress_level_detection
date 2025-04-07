"""
Database utilities for the stress level detection dashboard.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database connection URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create SQLAlchemy engine
if DATABASE_URL is not None:  # <-- This is the critical check
    engine = create_engine(DATABASE_URL)
    print("Using PostgreSQL database.")
else:
    # Use a SQLite database as fallback
    engine = create_engine('sqlite:///stress_analysis.db')
    print("Using SQLite database as fallback since DATABASE_URL is not provided.")

Base = declarative_base()
metadata = MetaData()

# Define the stress_data table
stress_data = Table(
    'stress_data', 
    metadata,
    Column('id', Integer, primary_key=True),
    Column('subreddit', String),
    Column('post_id', String),
    Column('text', String),
    Column('label', Integer),
    Column('confidence', Float),
    Column('social_timestamp', Integer),
    Column('social_karma', Integer),
    Column('sentiment', Float),
    # Add other columns as needed
)

# Create tables
def init_db():
    """
    Initialize the database by creating all tables.
    """
    try:
        metadata.create_all(engine)
        print("Database tables created successfully.")
    except Exception as e:
        print(f"Error creating database tables: {e}")

# Function to load CSV data into the database
def load_csv_to_db(csv_path):
    """
    Load data from a CSV file into the database.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the CSV file
        data = pd.read_csv(csv_path)
        
        # Select a subset of columns to store in the database
        # This can be modified to include more or fewer columns
        subset_cols = [
            'id', 'subreddit', 'post_id', 'text', 'label', 'confidence', 
            'social_timestamp', 'social_karma', 'sentiment', 'lex_liwc_anx', 
            'lex_liwc_anger', 'lex_liwc_sad', 'lex_liwc_negemo', 'lex_liwc_posemo'
        ]
        
        # Filter columns that exist in the dataframe
        valid_cols = [col for col in subset_cols if col in data.columns]
        subset_data = data[valid_cols]
        
        # Write to database
        subset_data.to_sql('stress_data', engine, if_exists='replace', index=False)
        print(f"Data loaded from {csv_path} and saved to database.")
        
        return True
    except Exception as e:
        print(f"Error loading data to database: {e}")
        return False

# Function to load data from the database
def load_data_from_db():
    """
    Load the stress data from the database.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Query all data from the stress_data table
        query = "SELECT * FROM stress_data"
        df = pd.read_sql(query, engine)
        print("Data loaded from database successfully.")
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

# Function to check if the stress_data table exists and has data
def check_data_exists():
    """
    Check if the stress_data table exists and has data.
    
    Returns:
        bool: True if the table exists and has data, False otherwise
    """
    try:
        # Use inspector to check if the table exists
        from sqlalchemy import inspect
        inspector = inspect(engine)
        
        if not inspector.has_table('stress_data'):
            print("Table 'stress_data' does not exist.")
            return False
        
        # Check if the table has data
        query = "SELECT COUNT(*) FROM stress_data"
        result = pd.read_sql(query, engine)
        count = result.iloc[0, 0]
        print(f"Table 'stress_data' has {count} records.")
        return count > 0
    except Exception as e:
        print(f"Error checking if data exists: {e}")
        return False

# Function to save model predictions to the database
def save_predictions(post_ids, predictions, probabilities):
    """
    Save model predictions to the database.
    
    Args:
        post_ids (list): List of post IDs
        predictions (list): List of predicted labels
        probabilities (list): List of prediction probabilities
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a dataframe from the predictions
        predictions_df = pd.DataFrame({
            'post_id': post_ids,
            'predicted_label': predictions,
            'probability': probabilities
        })
        
        # Write to database
        predictions_df.to_sql('predictions', engine, if_exists='replace', index=False)
        print("Predictions saved to database successfully.")
        
        return True
    except Exception as e:
        print(f"Error saving predictions to database: {e}")
        return False

# Initialize the database
init_db()