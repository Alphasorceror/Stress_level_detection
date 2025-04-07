
Stress Level Detection Dashboard Project Overview
This project is an interactive Streamlit dashboard designed to analyze and visualize stress levels in social media text data, primarily from Reddit. The application provides valuable insights into how stress manifests in online communications using data analysis, machine learning, and visualization techniques.

Core Features
Data Analysis & Exploration

Visualizes stress distribution in social media posts
Provides time-based analysis of stress indicators
Analyzes stress by different subreddit communities
Shows correlations between emotional features and stress
Predictive Modeling

Trains machine learning models to detect stress in text
Supports multiple model types (Random Forest, Logistic Regression, SVM, etc.)
Evaluates model performance with metrics like accuracy and confusion matrices
Provides feature importance analysis to understand what drives stress detection
Text Analysis

Generates word clouds for stressed and non-stressed content
Analyzes sentiment and emotional content in text
Identifies key linguistic patterns associated with stress
Database Integration

Uses PostgreSQL with SQLite fallback for data storage
Stores original dataset and model predictions
Provides database utilities for data loading and saving
Technical Stack
Frontend: Streamlit dashboard with multiple interactive sections
Data Processing: Pandas and NumPy for data manipulation
Visualization: Plotly, Matplotlib, Seaborn, and WordCloud
Machine Learning: Scikit-learn for predictive modeling
Database: SQLAlchemy with PostgreSQL/SQLite for data persistence
