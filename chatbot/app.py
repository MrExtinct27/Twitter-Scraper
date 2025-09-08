import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import re
from datetime import datetime
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Sentiment Tools ---
sentiment_analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    if pd.isna(text) or text == "":
        return 0
    return sentiment_analyzer.polarity_scores(str(text))['compound']

def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def extract_hashtags(text):
    if pd.isna(text):
        return []
    return re.findall(r'#\w+', str(text))

# --- Load Data ---
@st.cache_data
def load_data():
    csv_files = [
        "../social_media_posts_dakota_access_pipeline.csv",
        "../social_media_posts_concophillips_willow_project.csv",
        "../social_media_posts_texas_border_wall.csv"
    ]
    dataframes = []
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    df = pd.concat(dataframes, ignore_index=True).drop_duplicates(subset='post_id')
    df['content'] = df['content'].fillna('')
    df['posted_at'] = pd.to_datetime(df['posted_at'], errors='coerce')
    df['captured_at'] = pd.to_datetime(df['captured_at'], errors='coerce')
    df['hour'] = df['posted_at'].dt.hour
    df['day_of_week'] = df['posted_at'].dt.day_name()
    df['date'] = df['posted_at'].dt.date
    df['sentiment_score'] = df['content'].apply(get_sentiment_score)
    df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)
    df['content_length'] = df['content'].str.len()
    df['word_count'] = df['content'].str.split().str.len()
    df['hashtags'] = df['content'].apply(extract_hashtags)

    # Fix common keyword typos
    keyword_fix = {
        "ConcoPhillips Willow Project": "ConocoPhillips Willow Project",
        "Willow Project": "ConocoPhillips Willow Project",
        "Willow": "ConocoPhillips Willow Project",
        "ConocoPhillips": "ConocoPhillips Willow Project",
        "conocophillips willow": "ConocoPhillips Willow Project",
    }
    df['keyword'] = df['keyword'].replace(keyword_fix, regex=False)

    return df


# --- Streamlit App ---
st.set_page_config(page_title="Social Media Analyst Bot", layout="wide")
st.title("📊 Social Media Intelligence Analyst Bot")

with st.spinner("Loading data and processing..."):
    df = load_data()

if df.empty:
    st.warning("No data found. Ensure CSV files exist.")
    st.stop()

st.subheader("📁 Dataset Overview")
st.write(f"Total Posts: {len(df)}")
st.write(f"Date Range: {df['posted_at'].min()} to {df['posted_at'].max()}")
st.write(f"Keywords: {', '.join(sorted(df['keyword'].dropna().unique()))}")
st.write(df.head())

st.subheader("🤖 Ask a Question About the Data")
query = st.text_input("Enter your question:")

if query and GROQ_API_KEY:
    context_data = {
        'total_posts': len(df),
        'date_range': f"{df['posted_at'].min()} to {df['posted_at'].max()}",
        'unique_users': df['username'].nunique(),
        'sentiment_distribution': df['sentiment_label'].value_counts(normalize=True).mul(100).round(2).to_dict()
    }

    system_message = f"""
You are a Social Media Intelligence Analyst specialized in analyzing Twitter data about environmental and political topics.

You have access to scraped Twitter data about: Dakota Access Pipeline, ConocoPhillips Willow Project, and Texas Border Wall.

Current Data Context:
- Total Posts: {context_data['total_posts']:,}
- Date Range: {context_data['date_range']}
- Unique Users: {context_data['unique_users']:,}
- Sentiment Distribution: {context_data['sentiment_distribution']}

Instructions:
1. Provide insights based on the data shown above.
2. Include metrics, trends, and reference actual data.
3. Be objective and informative.
"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        result = response.choices[0].message.content
        st.markdown(f"**AI Analyst Response:**\n\n{result}")

    except Exception as e:
        st.error(f"Error from Groq API: {e}")

# Visual summary
st.subheader("Sentiment Distribution")
sentiment_counts = df['sentiment_label'].value_counts()
st.bar_chart(sentiment_counts)

st.subheader("Activity Over Time")
timeline = df.groupby('date').size()
st.line_chart(timeline)

st.subheader("Top Hashtags")
all_tags = [tag for sublist in df['hashtags'] for tag in sublist if isinstance(sublist, list)]
if all_tags:
    top_tags = pd.Series(all_tags).value_counts().head(10)
    st.bar_chart(top_tags)
else:
    st.info("No hashtags found in dataset.")

st.subheader("Sentiment Heatmap (Hour vs Day)")
heatmap_data = df.copy()
heatmap_data = heatmap_data.dropna(subset=["sentiment_score", "hour", "day_of_week"])

# Ensure correct day order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data['day_of_week'] = pd.Categorical(heatmap_data['day_of_week'], categories=day_order, ordered=True)

# Pivot for heatmap
pivot = heatmap_data.pivot_table(
    index='day_of_week',
    columns='hour',
    values='sentiment_score',
    aggfunc='mean'
)

# Plot with seaborn
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
plt.title("Average Sentiment Score by Day and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
st.pyplot(fig)
