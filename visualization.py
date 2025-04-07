"""
Visualization functions for the stress level detection dashboard.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from wordcloud import WordCloud
from utils import get_feature_categories

def plot_stress_distribution(df, label_column='label'):
    """
    Plot the distribution of stress vs. non-stress posts.
    
    Args:
        df (pd.DataFrame): Dataframe with stress data
        label_column (str): Name of the label column
        
    Returns:
        plotly.graph_objects.Figure: Distribution plot
    """
    if df is None or df.empty or label_column not in df.columns:
        return go.Figure()
    
    # Count the number of stress and non-stress posts
    label_counts = df[label_column].value_counts().reset_index()
    label_counts.columns = ['Stress Level', 'Count']
    
    # Map 0 to 'No Stress' and 1 to 'Stress'
    label_counts['Stress Level'] = label_counts['Stress Level'].map({0: 'No Stress', 1: 'Stress'})
    
    # Create a pie chart
    fig = px.pie(
        label_counts,
        values='Count',
        names='Stress Level',
        title='Distribution of Stress vs. Non-Stress Posts',
        color_discrete_sequence=['#3498db', '#e74c3c'],
        hole=0.3
    )
    
    fig.update_layout(
        legend_title_text='Stress Level',
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5)
    )
    
    return fig

def plot_subreddit_stress_rate(df):
    """
    Plot the stress rate by subreddit.
    
    Args:
        df (pd.DataFrame): DataFrame with subreddit stats
        
    Returns:
        plotly.graph_objects.Figure: Bar plot
    """
    if df is None or df.empty:
        return go.Figure()
    
    # Sort by stress rate
    df_sorted = df.sort_values('stress_rate', ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        df_sorted,
        x='subreddit',
        y='stress_rate',
        text=df_sorted['stress_rate'].round(1).astype(str) + '%',
        title='Stress Rate by Subreddit',
        color='stress_rate',
        color_continuous_scale='Reds',
        hover_data=['post_count', 'avg_anxiety', 'avg_negative_emotion']
    )
    
    fig.update_layout(
        xaxis_title='Subreddit',
        yaxis_title='Stress Rate (%)',
        yaxis=dict(ticksuffix='%'),
        coloraxis_showscale=False
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def plot_feature_correlation(df, features, target='label'):
    """
    Plot correlation heatmap for selected features.
    
    Args:
        df (pd.DataFrame): Dataframe with features
        features (list): List of feature names
        target (str): Target variable name
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    if df is None or df.empty:
        return go.Figure()
    
    # Ensure all features exist in the dataframe
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        return go.Figure()
    
    # Include target in correlation matrix
    if target in df.columns:
        cols = valid_features + [target]
    else:
        cols = valid_features
    
    # Calculate correlation matrix
    corr_matrix = df[cols].corr()
    
    # Create a heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Heatmap',
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        height=600,
        width=700,
        xaxis=dict(tickangle=-45)
    )
    
    return fig

def plot_time_series(df, feature, time_col='date', group_by='month'):
    """
    Plot time series of a feature.
    
    Args:
        df (pd.DataFrame): Dataframe with time data
        feature (str): Feature to plot
        time_col (str): Time column name
        group_by (str): How to group the time (day, month, year)
        
    Returns:
        plotly.graph_objects.Figure: Time series plot
    """
    if df is None or df.empty or feature not in df.columns or time_col not in df.columns:
        return go.Figure()
    
    # Create a completely fresh copy of the dataframe with only needed columns
    df_subset = df[[time_col, feature]].copy()
    
    # Ensure the time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_subset[time_col]):
        try:
            df_subset[time_col] = pd.to_datetime(df_subset[time_col])
        except:
            return go.Figure()
    
    # Different approach based on grouping time period
    if group_by == 'day':
        # Extract date string
        df_subset['time_group'] = df_subset[time_col].dt.strftime('%Y-%m-%d')
    elif group_by == 'week':
        # Format year and week, avoiding isocalendar() which might not be available in older pandas
        try:
            df_subset['time_group'] = df_subset[time_col].apply(
                lambda x: f"{x.year}-W{x.week}" if hasattr(x, 'week') else f"{x.year}-W{x.strftime('%U')}"
            )
        except:
            # Simple fallback
            df_subset['time_group'] = df_subset[time_col].dt.strftime('%Y-%U')
    elif group_by == 'month':
        # Format as year-month
        df_subset['time_group'] = df_subset[time_col].dt.strftime('%Y-%m')
    else:  # year
        df_subset['time_group'] = df_subset[time_col].dt.strftime('%Y')
    
    # Group by the time_group and calculate mean of the feature
    result = df_subset.groupby('time_group')[feature].mean().reset_index()
    
    # Create line plot
    fig = px.line(
        result,
        x='time_group',
        y=feature,
        title=f'Time Series of {feature} ({group_by.title()})',
        markers=True
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title=feature
    )
    
    return fig

def plot_anxiety_vs_sentiment(df):
    """
    Plot anxiety score vs. sentiment.
    
    Args:
        df (pd.DataFrame): Dataframe with anxiety and sentiment scores
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot
    """
    if df is None or df.empty or 'lex_liwc_anx' not in df.columns or 'sentiment' not in df.columns:
        return go.Figure()
    
    # Create a scatter plot
    fig = px.scatter(
        df,
        x='lex_liwc_anx',
        y='sentiment',
        color='label',
        color_discrete_sequence=['#3498db', '#e74c3c'],
        title='Anxiety Score vs. Sentiment',
        hover_data=['subreddit'],
        opacity=0.7
    )
    
    # Add vertical and horizontal lines at zero
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        xaxis_title='Anxiety Score',
        yaxis_title='Sentiment',
        legend_title='Stress Level',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Update color mapping
    fig.update_traces(marker=dict(size=8))
    
    return fig

def generate_wordcloud(df, text_column='clean_text', label_column='label', label_value=1):
    """
    Generate wordcloud for text data.
    
    Args:
        df (pd.DataFrame): Dataframe with text data
        text_column (str): Name of the text column
        label_column (str): Name of the label column
        label_value (int): Value of label to filter by (0 or 1)
        
    Returns:
        bytes: Image bytes of the wordcloud or None if not enough data
    """
    if df is None or df.empty or text_column not in df.columns:
        return None
    
    # Filter text by label if specified
    if label_column in df.columns:
        filtered_df = df[df[label_column] == label_value]
    else:
        filtered_df = df
    
    if filtered_df.empty:
        return None
    
    # Combine all text and filter out non-text content
    all_text = ' '.join([str(text) for text in filtered_df[text_column].fillna('') if str(text).strip()])
    
    # Additional check for empty text
    if not all_text or len(all_text.strip()) < 10:  # Require at least some meaningful content
        # Create a simple message image instead of a wordcloud
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, "Not enough text data to generate wordcloud", 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        
        # Save figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Close the figure to free memory
        plt.close()
        
        return buf
    
    try:
        # Generate wordcloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=200,
            contour_width=1,
            contour_color='steelblue',
            min_word_length=3  # Ignore very short words
        ).generate(all_text)
        
        # Create figure
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Save figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Close the figure to free memory
        plt.close()
        
        return buf
    except Exception as e:
        # Create an error message image
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Unable to generate wordcloud: {str(e)}", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        # Save figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Close the figure to free memory
        plt.close()
        
        return buf

def plot_feature_category_importance(importance_df):
    """
    Plot importance of feature categories.
    
    Args:
        importance_df (pd.DataFrame): Dataframe with feature importance scores and categories
        
    Returns:
        plotly.graph_objects.Figure: Bar plot
    """
    if importance_df is None or importance_df.empty or 'Category' not in importance_df.columns:
        return go.Figure()
    
    # Group by category and calculate mean importance
    category_importance = importance_df.groupby('Category')['Importance'].mean().reset_index()
    
    # Sort by importance
    category_importance = category_importance.sort_values('Importance', ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        category_importance,
        x='Category',
        y='Importance',
        title='Average Feature Importance by Category',
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title='Feature Category',
        yaxis_title='Average Importance',
        coloraxis_showscale=False
    )
    
    return fig

def plot_stress_by_day_of_week(df):
    """
    Plot stress rate by day of week.
    
    Args:
        df (pd.DataFrame): Dataframe with date and stress data
        
    Returns:
        plotly.graph_objects.Figure: Bar plot
    """
    if df is None or df.empty or 'day_of_week' not in df.columns or 'label' not in df.columns:
        return go.Figure()
    
    # Map day of week numbers to names
    day_map = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    
    # Create a new column with day names
    df_copy = df.copy()
    df_copy['day_name'] = df_copy['day_of_week'].map(day_map)
    
    # Group by day of week and calculate stress rate
    day_stress = df_copy.groupby('day_name').agg(
        stress_rate=('label', 'mean'),
        count=('label', 'count')
    ).reset_index()
    
    # Convert stress rate to percentage
    day_stress['stress_rate'] = day_stress['stress_rate'] * 100
    
    # Set order of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stress['day_name'] = pd.Categorical(day_stress['day_name'], categories=day_order, ordered=True)
    day_stress = day_stress.sort_values('day_name')
    
    # Create a bar chart
    fig = px.bar(
        day_stress,
        x='day_name',
        y='stress_rate',
        text=day_stress['stress_rate'].round(1).astype(str) + '%',
        title='Stress Rate by Day of Week',
        color='stress_rate',
        color_continuous_scale='Reds',
        hover_data=['count']
    )
    
    fig.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Stress Rate (%)',
        yaxis=dict(ticksuffix='%'),
        coloraxis_showscale=False
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def plot_subreddit_emotion_radar(df, subreddit_list):
    """
    Generate radar chart for emotional features by subreddit.
    
    Args:
        df (pd.DataFrame): Dataframe with emotional features
        subreddit_list (list): List of subreddits to include
        
    Returns:
        plotly.graph_objects.Figure: Radar chart
    """
    if df is None or df.empty or 'subreddit' not in df.columns:
        return go.Figure()
    
    # Emotional features to plot
    emotion_features = ['lex_liwc_anx', 'lex_liwc_anger', 'lex_liwc_sad', 
                        'lex_liwc_posemo', 'lex_liwc_negemo']
    
    # Check if all features exist
    emotion_features = [f for f in emotion_features if f in df.columns]
    
    if not emotion_features:
        return go.Figure()
    
    # Filter by subreddit
    filtered_df = df[df['subreddit'].isin(subreddit_list)]
    
    if filtered_df.empty:
        return go.Figure()
    
    # Group by subreddit and calculate mean of emotion features
    grouped_df = filtered_df.groupby('subreddit')[emotion_features].mean().reset_index()
    
    # Create radar chart
    fig = go.Figure()
    
    # Add a trace for each subreddit
    for subreddit in grouped_df['subreddit']:
        subreddit_data = grouped_df[grouped_df['subreddit'] == subreddit]
        
        # Format feature names for display
        feature_names = [f.replace('lex_liwc_', '') for f in emotion_features]
        
        # Add trace
        fig.add_trace(go.Scatterpolar(
            r=subreddit_data[emotion_features].values[0],
            theta=feature_names,
            fill='toself',
            name=subreddit
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, grouped_df[emotion_features].max().max() * 1.1]
            )
        ),
        title='Emotional Features by Subreddit',
        showlegend=True
    )
    
    return fig