"""
Streamlit dashboard for stress level detection from social media text.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Import custom modules
from utils import (
    load_data, get_feature_categories, convert_timestamp_to_date, 
    get_stress_indicators, get_subreddit_stats, scale_features
)
from preprocessing import (
    prepare_data, select_features, split_dataset, get_feature_importance_df
)
from modeling import (
    train_model, evaluate_model, get_feature_importance, 
    plot_confusion_matrix, plot_feature_importance, predict_stress,
    cross_validate_model
)
from visualization import (
    plot_stress_distribution, plot_subreddit_stress_rate, plot_feature_correlation,
    plot_time_series, plot_anxiety_vs_sentiment, generate_wordcloud,
    plot_feature_category_importance, plot_stress_by_day_of_week,
    plot_subreddit_emotion_radar
)

# Set page configuration
st.set_page_config(
    page_title="Stress Level Detection Dashboard",
    page_icon="😓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Stress Level Detection Dashboard")
st.markdown("""
This interactive dashboard analyzes stress levels in social media text data.
Explore key indicators, patterns, and predictive models to understand stress manifestation in online communications.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Data Exploration", "Feature Analysis", "Predictive Modeling", "Text Analysis", "About"]
)

# Import database functions
import database

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    # Use database by default, with fallback to CSV
    df = load_data("stress.csv", use_db=True)
    if df is not None:
        prepared_df = prepare_data(df)
        return df, prepared_df
    return None, None

data_load_state = st.sidebar.text("Loading data...")
df, prepared_df = load_and_prepare_data()
data_load_state.text("Data loaded successfully!" if df is not None else "Error loading data!")

if df is None:
    st.error("Failed to load the dataset. Please check if the file 'stress.csv' exists in the current directory.")
    st.stop()

# Feature selection
@st.cache_data
def get_selected_features(_df):
    selected_features, feature_scores = select_features(_df, k=20)
    importance_df = get_feature_importance_df(feature_scores)
    return selected_features, feature_scores, importance_df

selected_features, feature_scores, importance_df = get_selected_features(prepared_df)

# Overview page
if page == "Overview":
    st.header("Overview of Stress in Social Media Data")
    
    # Key metrics
    stress_indicators = get_stress_indicators(df)
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Stress Prevalence", 
            f"{stress_indicators.get('Stress Prevalence', 0):.1f}%", 
            help="Percentage of posts classified as showing stress"
        )
    
    with col2:
        st.metric(
            "Anxiety Score (Avg)", 
            f"{stress_indicators.get('Anxiety Score (Avg)', 0):.2f}", 
            help="Average anxiety score across all posts"
        )
    
    with col3:
        st.metric(
            "Negative Emotion (Avg)", 
            f"{stress_indicators.get('Negative Emotion (Avg)', 0):.2f}", 
            help="Average negative emotion score across all posts"
        )
    
    with col4:
        st.metric(
            "Average Sentiment", 
            f"{stress_indicators.get('Average Sentiment', 0):.2f}", 
            delta=None,
            help="Average sentiment score (-1 to 1) across all posts"
        )
    
    # Distribution of stress vs. non-stress
    st.subheader("Stress Distribution")
    stress_dist_fig = plot_stress_distribution(df)
    st.plotly_chart(stress_dist_fig, use_container_width=True)
    
    # Stress rate by subreddit
    st.subheader("Stress Rate by Subreddit")
    subreddit_stats = get_subreddit_stats(df)
    subreddit_fig = plot_subreddit_stress_rate(subreddit_stats)
    st.plotly_chart(subreddit_fig, use_container_width=True)
    
    # Time-based stress rate
    st.subheader("Stress Rate Over Time")
    
    # Add date column if not already present
    if 'date' not in prepared_df.columns:
        time_df = convert_timestamp_to_date(prepared_df)
    else:
        time_df = prepared_df
        
    # Time grouping option
    time_group = st.radio(
        "Group by:",
        ["month", "week", "day"],
        horizontal=True
    )
    
    # Plot time series
    time_fig = plot_time_series(time_df, 'label', group_by=time_group)
    st.plotly_chart(time_fig, use_container_width=True)
    
    # Anxiety vs. Sentiment
    st.subheader("Anxiety vs. Sentiment")
    anxiety_sent_fig = plot_anxiety_vs_sentiment(prepared_df)
    st.plotly_chart(anxiety_sent_fig, use_container_width=True)
    
    # Day of week pattern
    st.subheader("Stress Patterns by Day of Week")
    dow_fig = plot_stress_by_day_of_week(prepared_df)
    st.plotly_chart(dow_fig, use_container_width=True)

# Data Exploration page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    # Dataset info
    st.subheader("Dataset Information")
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1]}")
    
    # Display subreddit distribution
    st.subheader("Subreddit Distribution")
    subreddit_counts = df['subreddit'].value_counts().reset_index()
    subreddit_counts.columns = ['Subreddit', 'Count']
    
    # Plot subreddit distribution
    fig = px.bar(
        subreddit_counts, 
        x='Subreddit', 
        y='Count',
        color='Count',
        color_continuous_scale='Viridis',
        title='Number of Posts by Subreddit'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show sample data
    st.subheader("Sample Data")
    with st.expander("Click to view sample data"):
        sample_size = st.slider("Sample size", 5, 50, 10)
        st.dataframe(df.sample(sample_size, random_state=42))
    
    # Feature statistics
    st.subheader("Feature Statistics")
    feature_categories = get_feature_categories()
    
    # Select feature category
    category = st.selectbox(
        "Select a feature category:",
        list(feature_categories.keys())
    )
    
    # Filter features by category
    category_features = feature_categories[category]
    valid_features = [f for f in category_features if f in df.columns]
    
    if valid_features:
        # Display statistics
        stats_df = df[valid_features].describe().T
        st.dataframe(stats_df)
        
        # Feature histograms
        st.subheader(f"Distribution of {category} Features")
        
        # Select feature for histogram
        feature = st.selectbox("Select a feature:", valid_features)
        
        # Plot histogram
        hist_fig = px.histogram(
            df, 
            x=feature,
            color='label',
            marginal='box',
            title=f'Distribution of {feature}',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.warning(f"No features found in the '{category}' category.")
    
    # Correlations between selected features
    st.subheader("Feature Correlations")
    
    # Select features for correlation
    corr_features = st.multiselect(
        "Select features for correlation analysis:",
        selected_features,
        default=selected_features[:5] if len(selected_features) >= 5 else selected_features
    )
    
    if corr_features:
        corr_fig = plot_feature_correlation(prepared_df, corr_features)
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.warning("Please select at least one feature for correlation analysis.")

# Feature Analysis page
elif page == "Feature Analysis":
    st.header("Feature Analysis")
    
    # Top important features
    st.subheader("Top Important Features for Stress Detection")
    
    # Display feature importance dataframe
    if not importance_df.empty:
        st.dataframe(
            importance_df.head(20)[['Feature', 'Importance', 'Category']]
            .style.background_gradient(subset=['Importance'], cmap='viridis')
        )
        
        # Plot feature importance
        top_n = st.slider("Number of top features to display", 5, 20, 10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot top features
            feature_imp_fig = plot_feature_category_importance(importance_df)
            st.plotly_chart(feature_imp_fig, use_container_width=True)
        
        with col2:
            # Individual feature importance
            if feature_scores:
                imp_buf = plot_feature_importance(feature_scores, top_n=top_n)
                
                if imp_buf:
                    st.image(imp_buf)
                else:
                    st.warning("Could not generate feature importance plot.")
            else:
                st.warning("Feature importance scores not available.")
    else:
        st.warning("Feature importance data not available.")
    
    # Feature comparisons
    st.subheader("Feature Comparison by Stress Level")
    
    # Select features to compare
    compare_x = st.selectbox(
        "Select feature for X-axis:",
        selected_features,
        index=0 if selected_features else 0
    )
    
    compare_y = st.selectbox(
        "Select feature for Y-axis:",
        selected_features,
        index=min(1, len(selected_features)-1) if len(selected_features) > 1 else 0
    )
    
    if compare_x and compare_y:
        # Create scatter plot
        fig = px.scatter(
            prepared_df, 
            x=compare_x, 
            y=compare_y,
            color='label',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            title=f'{compare_x} vs {compare_y} by Stress Level',
            opacity=0.7,
            hover_data=['subreddit']
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=compare_x,
            yaxis_title=compare_y,
            legend_title='Stress Level'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select features for comparison.")
    
    # Subreddit feature analysis
    st.subheader("Emotional Features by Subreddit")
    
    # Select subreddits
    subreddit_options = df['subreddit'].unique().tolist()
    selected_subreddits = st.multiselect(
        "Select subreddits to compare:",
        subreddit_options,
        default=subreddit_options[:3] if len(subreddit_options) >= 3 else subreddit_options
    )
    
    if selected_subreddits:
        # Generate radar chart
        radar_fig = plot_subreddit_emotion_radar(prepared_df, selected_subreddits)
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.warning("Please select at least one subreddit for comparison.")

# Predictive Modeling page
elif page == "Predictive Modeling":
    st.header("Predictive Modeling for Stress Detection")
    
    # Model selection
    st.subheader("Model Configuration")
    
    model_type = st.selectbox(
        "Select model type:",
        ["random_forest", "logistic", "gradient_boosting", "svm"]
    )
    
    # Feature selection for model
    st.subheader("Feature Selection")
    
    if not selected_features:
        st.warning("No features selected. Please check feature selection process.")
        st.stop()
    
    model_features = st.multiselect(
        "Select features for the model:",
        selected_features,
        default=selected_features[:10] if len(selected_features) >= 10 else selected_features
    )
    
    if not model_features:
        st.warning("Please select at least one feature for the model.")
        st.stop()
    
    # Train and evaluate model
    train_button = st.button("Train Model")
    
    if train_button:
        with st.spinner("Training model..."):
            # Split the dataset
            X_train, X_test, y_train, y_test = split_dataset(prepared_df, model_features)
            
            if X_train is None:
                st.error("Error splitting dataset. Please check your feature selection.")
                st.stop()
            
            # Train the model
            model = train_model(X_train, y_train, model_type=model_type)
            
            if model is None:
                st.error("Error training model. Please try a different model type or features.")
                st.stop()
            
            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test)
            
            if not metrics:
                st.error("Error evaluating model. Please try again.")
                st.stop()
            
            # Display metrics
            st.subheader("Model Performance")
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
            
            with col2:
                st.metric("Precision", f"{metrics['precision']:.2f}")
            
            with col3:
                st.metric("Recall", f"{metrics['recall']:.2f}")
            
            with col4:
                st.metric("F1 Score", f"{metrics['f1']:.2f}")
            
            # Cross-validation
            cv_mean, cv_std = cross_validate_model(
                X_train, y_train, model_type=model_type, cv=5
            )
            
            st.info(f"**Cross-validation F1 Score:** {cv_mean:.2f} ± {cv_std:.2f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm_buf = plot_confusion_matrix(model, X_test, y_test)
            
            if cm_buf:
                st.image(cm_buf)
            else:
                st.warning("Could not generate confusion matrix.")
            
            # Feature importance
            st.subheader("Model Feature Importance")
            model_feature_importance = get_feature_importance(model, model_features)
            
            if model_feature_importance:
                # Plot feature importance
                fi_buf = plot_feature_importance(model_feature_importance)
                
                if fi_buf:
                    st.image(fi_buf)
                else:
                    st.warning("Could not generate feature importance plot.")
            else:
                st.warning("Feature importance not available for this model type.")
    
    # Stress prediction on sample posts
    st.subheader("Predict Stress on Sample Posts")
    
    # Get a random sample for prediction
    if st.button("Load Random Sample"):
        # Select a random sample from the dataset
        sample_idx = np.random.randint(0, len(df))
        sample_post = df.iloc[sample_idx]
        
        # Display the post
        st.text_area("Post Text:", value=sample_post['text'], height=150)
        
        # Prepare features for prediction
        sample_features = prepared_df.iloc[sample_idx][model_features].values.reshape(1, -1)
        
        # Train model if not already trained
        if 'model' not in locals():
            with st.spinner("Training model for prediction..."):
                # Split the dataset
                X_train, X_test, y_train, y_test = split_dataset(prepared_df, model_features)
                
                if X_train is None:
                    st.error("Error splitting dataset. Please check your feature selection.")
                    st.stop()
                
                # Train the model
                model = train_model(X_train, y_train, model_type=model_type)
        
        # Make prediction
        pred_label, pred_prob = predict_stress(model, sample_features)
        
        # Save prediction to database
        if pred_label is not None and pred_prob is not None:
            # Get the post ID
            post_id = sample_post.get('post_id', str(sample_idx))
            # Save to database
            database.save_predictions([post_id], [int(pred_label)], [float(pred_prob)])
            st.success("Prediction saved to database!")
        
        # Display prediction
        if pred_label is not None:
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Predicted Label", 
                    "Stress" if pred_label == 1 else "No Stress",
                    delta=None
                )
            
            with col2:
                if pred_prob is not None:
                    st.metric(
                        "Confidence", 
                        f"{pred_prob:.2f}" if pred_label == 1 else f"{1-pred_prob:.2f}",
                        delta=None
                    )
            
            # Display actual label
            st.info(f"Actual Label: {'Stress' if sample_post['label'] == 1 else 'No Stress'}")

# Text Analysis page
elif page == "Text Analysis":
    st.header("Text Analysis")
    
    # Word clouds
    st.subheader("Word Clouds")
    
    # Create tabs for stress and non-stress word clouds
    wc_tab1, wc_tab2 = st.tabs(["Stress Posts", "Non-Stress Posts"])
    
    with wc_tab1:
        st.write("Word cloud for posts labeled as stress:")
        stress_wc = generate_wordcloud(prepared_df, text_column='clean_text', label_value=1)
        
        if stress_wc:
            st.image(stress_wc)
        else:
            st.warning("Could not generate word cloud for stress posts.")
    
    with wc_tab2:
        st.write("Word cloud for posts labeled as non-stress:")
        nonstress_wc = generate_wordcloud(prepared_df, text_column='clean_text', label_value=0)
        
        if nonstress_wc:
            st.image(nonstress_wc)
        else:
            st.warning("Could not generate word cloud for non-stress posts.")
    
    # Text content analysis
    st.subheader("Text Content Analysis")
    
    # Show a sample post
    st.write("Analyze a sample post:")
    
    # Get a random sample
    sample_idx = np.random.randint(0, len(df))
    sample_post = df.iloc[sample_idx]
    
    # Display the post
    st.text_area("Sample Post:", value=sample_post['text'], height=150)
    
    # Display post metadata
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Subreddit:** {sample_post['subreddit']}")
        st.write(f"**Label:** {'Stress' if sample_post['label'] == 1 else 'No Stress'}")
    
    with col2:
        if 'sentiment' in sample_post:
            st.write(f"**Sentiment:** {sample_post['sentiment']:.2f}")
        
        if 'lex_liwc_anx' in sample_post:
            st.write(f"**Anxiety Score:** {sample_post['lex_liwc_anx']:.2f}")
    
    # Show emotional features
    st.subheader("Emotional Features")
    
    emotion_cols = [
        'lex_liwc_anx', 'lex_liwc_anger', 'lex_liwc_sad', 
        'lex_liwc_posemo', 'lex_liwc_negemo'
    ]
    
    # Filter valid columns
    valid_emotion_cols = [col for col in emotion_cols if col in sample_post.index]
    
    if valid_emotion_cols:
        # Create a bar chart
        emotion_data = pd.DataFrame({
            'Emotion': [col.replace('lex_liwc_', '').title() for col in valid_emotion_cols],
            'Score': [sample_post[col] for col in valid_emotion_cols]
        })
        
        fig = px.bar(
            emotion_data, 
            x='Emotion', 
            y='Score',
            title="Emotional Features in Sample Post",
            color='Score',
            color_continuous_scale='RdBu_r'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Emotional feature data not available for this post.")

# About page
elif page == "About":
    st.header("About the Stress Level Detection Dashboard")
    
    st.write("""
    This dashboard analyzes stress levels in social media text data, focusing on identifying 
    patterns, key indicators, and building predictive models for stress detection.
    """)
    
    st.subheader("Dataset Information")
    
    st.write("""
    The dataset contains social media posts from Reddit, labeled as stress (1) or non-stress (0).
    Each post includes various features extracted using linguistic analysis, including:
    
    - **Social metrics**: timestamp, karma, upvote ratio, etc.
    - **Linguistic features**: LIWC (Linguistic Inquiry and Word Count) analysis
    - **Sentiment scores**: Overall sentiment and emotion indicators
    - **Text content**: The original post text
    """)
    
    st.subheader("Dataset Statistics")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total Posts:** {df.shape[0]}")
        st.write(f"**Subreddits:** {df['subreddit'].nunique()}")
        st.write(f"**Features:** {df.shape[1]}")
    
    with col2:
        st.write(f"**Stress Posts:** {df[df['label'] == 1].shape[0]} ({df['label'].mean() * 100:.1f}%)")
        st.write(f"**Non-Stress Posts:** {df[df['label'] == 0].shape[0]} ({(1 - df['label'].mean()) * 100:.1f}%)")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Features")
with st.sidebar.expander("View Feature Categories"):
    feature_categories = get_feature_categories()
    for category, features in feature_categories.items():
        st.write(f"**{category}**")
        st.write(", ".join(features))

st.sidebar.markdown("---")
st.sidebar.markdown("© 2023 Stress Level Detection Dashboard")