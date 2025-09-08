import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# AI imports - Simplified approach
import openai
from groq import Groq
import json

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Environment variables
from dotenv import load_dotenv
load_dotenv()

class SimpleSocialMediaAnalyst:
    def __init__(self):
        """Initialize the Social Media Intelligence Analyst Bot"""
        print("Initializing Social Media Intelligence Analyst Bot...")
        
        # Load environment variables
        # Add these to your .env file:
        # GROQ_API_KEY=your_groq_api_key_here
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.groq_api_key)
        
        # Initialize components
        self.df = None
        self.processed_data = None
        self.chat_history = []
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        print("Bot initialized successfully!")

    def load_data(self, csv_files):
        """Load and combine CSV data from multiple files"""
        print(f"Loading data from {len(csv_files)} CSV files...")
        
        dataframes = []
        for file in csv_files:
            if os.path.exists(file):
                df_temp = pd.read_csv(file)
                dataframes.append(df_temp)
                print(f"   Loaded {len(df_temp)} posts from {file}")
            else:
                print(f"   File not found: {file}")
        
        if not dataframes:
            raise ValueError("No valid CSV files found!")
        
        # Combine all dataframes
        self.df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['post_id'], keep='first')
        final_count = len(self.df)
        
        print(f"Combined dataset: {final_count} unique posts ({initial_count - final_count} duplicates removed)")
        
        # Process the data
        self._process_data()
        
        return self.df

    def _process_data(self):
        """Process and enrich the data with analytics"""
        print("Processing and analyzing data...")
        
        # Convert timestamps
        self.df['posted_at'] = pd.to_datetime(self.df['posted_at'], errors='coerce')
        self.df['captured_at'] = pd.to_datetime(self.df['captured_at'], errors='coerce')
        
        # Add time-based features
        self.df['hour'] = self.df['posted_at'].dt.hour
        self.df['day_of_week'] = self.df['posted_at'].dt.day_name()
        self.df['date'] = self.df['posted_at'].dt.date
        
        # Sentiment Analysis
        print("   Analyzing sentiment...")
        self.df['sentiment_score'] = self.df['content'].apply(self._get_sentiment_score)
        self.df['sentiment_label'] = self.df['sentiment_score'].apply(self._classify_sentiment)
        
        # Content analysis
        self.df['content_length'] = self.df['content'].str.len()
        self.df['word_count'] = self.df['content'].str.split().str.len()
        
        # Hashtag extraction
        self.df['hashtags'] = self.df['content'].apply(self._extract_hashtags)
        
        print(f"   Data processed successfully!")
        print(f"   Sentiment distribution: {self.df['sentiment_label'].value_counts().to_dict()}")

    def _get_sentiment_score(self, text):
        """Get sentiment score using VADER"""
        if pd.isna(text) or text == "":
            return 0
        return self.sentiment_analyzer.polarity_scores(str(text))['compound']

    def _classify_sentiment(self, score):
        """Classify sentiment based on score"""
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def _extract_hashtags(self, text):
        """Extract hashtags from text"""
        if pd.isna(text):
            return []
        return re.findall(r'#\w+', str(text))

    def get_data_context(self, query, limit=10):
        """Get relevant data context for the query"""
        context_data = {}
        
        # Basic statistics
        context_data['total_posts'] = len(self.df)
        context_data['date_range'] = f"{self.df['posted_at'].min()} to {self.df['posted_at'].max()}"
        context_data['unique_users'] = self.df['username'].nunique()
        
        # Sentiment distribution
        sentiment_dist = self.df['sentiment_label'].value_counts(normalize=True) * 100
        context_data['sentiment_distribution'] = sentiment_dist.to_dict()
        
        # Check if query mentions specific keywords
        query_lower = query.lower()
        relevant_keywords = []
        for keyword in self.df['keyword'].unique():
            if keyword.lower() in query_lower:
                relevant_keywords.append(keyword)
        
        if relevant_keywords:
            # Filter data for relevant keywords
            filtered_df = self.df[self.df['keyword'].isin(relevant_keywords)]
            context_data['filtered_posts'] = len(filtered_df)
            context_data['filtered_sentiment'] = filtered_df['sentiment_label'].value_counts(normalize=True).mul(100).to_dict()
            
            # Get sample posts
            sample_posts = filtered_df.head(limit)[['username', 'content', 'sentiment_label', 'posted_at']].to_dict('records')
            context_data['sample_posts'] = sample_posts
        
        # Check for specific analysis requests
        if any(word in query_lower for word in ['user', 'active', 'top']):
            top_users = self.df['username'].value_counts().head(10).to_dict()
            context_data['top_users'] = top_users
        
        if any(word in query_lower for word in ['hashtag', 'tag', 'trending']):
            all_hashtags = [tag for hashtag_list in self.df['hashtags'] for tag in hashtag_list]
            if all_hashtags:
                top_hashtags = Counter(all_hashtags).most_common(10)
                context_data['top_hashtags'] = dict(top_hashtags)
        
        if any(word in query_lower for word in ['time', 'when', 'hour', 'day']):
            hourly_activity = self.df['hour'].value_counts().sort_index().to_dict()
            daily_activity = self.df['day_of_week'].value_counts().to_dict()
            context_data['hourly_activity'] = hourly_activity
            context_data['daily_activity'] = daily_activity
        
        return context_data

    def ask_question(self, question: str):
        """Ask a question to the AI analyst"""
        print(f"\nQuestion: {question}")
        print("Analyzing data and generating response...")
        
        try:
            # Get relevant data context
            context = self.get_data_context(question)
            
            # Create system message with data context
            system_message = f"""
You are a Social Media Intelligence Analyst specialized in analyzing Twitter data about environmental and political topics.

You have access to scraped Twitter data about: Dakota Access Pipeline, ConocoPhillips Willow Project, and Texas Border Wall.

Current Data Context:
- Total Posts: {context['total_posts']:,}
- Date Range: {context['date_range']}
- Unique Users: {context['unique_users']:,}
- Sentiment Distribution: {context['sentiment_distribution']}

Additional Context: {json.dumps(context, indent=2, default=str)}

Instructions:
1. Provide analytical insights based on the data shown above
2. Include specific metrics, trends, and patterns when possible
3. Reference actual data points when relevant
4. Suggest visualizations when appropriate
5. Be objective and balanced in your analysis
6. If you don't have enough data, say so clearly

Respond as an expert data analyst would, with insights and actionable intelligence.
"""
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": question})
            
            # Create messages for API
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ]
            
            # Add recent chat history
            if len(self.chat_history) > 1:
                for msg in self.chat_history[-4:]:  # Last 4 messages for context
                    messages.append(msg)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # or "llama3-8b-8192" for faster responses
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # Add response to chat history
            self.chat_history.append({"role": "assistant", "content": answer})
            
            print(f"\nAnalysis:\n{answer}")
            
            return answer
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"{error_msg}")
            return error_msg

    def create_visualizations(self, topic='overall'):
        """Create visualizations for the data"""
        print(f"Creating visualizations for: {topic}")
        
        # Filter data if specific topic requested
        if topic.lower() != 'overall':
            plot_df = self.df[self.df['keyword'].str.contains(topic, case=False, na=False)]
            if len(plot_df) == 0:
                print(f"No data found for topic: {topic}")
                return
        else:
            plot_df = self.df
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Social Media Analysis Dashboard - {topic.title()}', fontsize=16)
        
        # 1. Sentiment Distribution
        sentiment_counts = plot_df['sentiment_label'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0, 0].set_title('Sentiment Distribution')
        
        # 2. Posts over Time
        daily_posts = plot_df.groupby('date').size().reset_index()
        daily_posts.columns = ['date', 'count']
        axes[0, 1].plot(daily_posts['date'], daily_posts['count'], marker='o')
        axes[0, 1].set_title('Posts Over Time')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Top Users
        top_users = plot_df['username'].value_counts().head(10)
        axes[1, 0].barh(range(len(top_users)), top_users.values)
        axes[1, 0].set_yticks(range(len(top_users)))
        axes[1, 0].set_yticklabels([f"@{user}" for user in top_users.index])
        axes[1, 0].set_title('Most Active Users')
        
        # 4. Hourly Activity
        hourly_activity = plot_df['hour'].value_counts().sort_index()
        axes[1, 1].bar(hourly_activity.index, hourly_activity.values, color='#3498db')
        axes[1, 1].set_title('Activity by Hour of Day')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Number of Posts')
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"social_media_analysis_{topic.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved as: {filename}")
        
        plt.show()

    def generate_report(self, topic='overall'):
        """Generate a comprehensive analysis report"""
        print(f"Generating comprehensive report for: {topic}")
        
        # Filter data if needed
        if topic.lower() != 'overall':
            report_df = self.df[self.df['keyword'].str.contains(topic, case=False, na=False)]
            if len(report_df) == 0:
                return f"No data found for topic: {topic}"
        else:
            report_df = self.df
        
        # Get analytics
        sentiment_stats = self.get_sentiment_analysis(topic if topic != 'overall' else None)
        user_stats = self.get_user_analysis(topic if topic != 'overall' else None)
        
        # Generate report
        report = f"""
# Social Media Intelligence Report: {topic.title()}

## Executive Summary
- **Total Posts Analyzed**: {len(report_df):,}
- **Date Range**: {report_df['posted_at'].min().strftime('%Y-%m-%d')} to {report_df['posted_at'].max().strftime('%Y-%m-%d')}
- **Unique Users**: {report_df['username'].nunique():,}
- **Average Posts per Day**: {len(report_df) / max(1, (report_df['posted_at'].max() - report_df['posted_at'].min()).days):.1f}

## Sentiment Analysis
- **Overall Sentiment Score**: {report_df['sentiment_score'].mean():.3f}
- **Sentiment Distribution**:
{report_df['sentiment_label'].value_counts(normalize=True).mul(100).round(1).apply(lambda x: f"  - {x:.1f}%").to_string()}

## User Engagement
- **Most Active Users**:
{report_df['username'].value_counts().head(5).apply(lambda x: f"  - {x} posts").to_string()}

## Content Insights
- **Average Content Length**: {report_df['content_length'].mean():.0f} characters
- **Average Word Count**: {report_df['word_count'].mean():.0f} words
- **Peak Activity Hour**: {report_df['hour'].value_counts().index[0]}:00
- **Most Active Day**: {report_df['day_of_week'].value_counts().index[0]}

## Top Hashtags
{pd.Series([tag for hashtags in report_df['hashtags'] for tag in hashtags]).value_counts().head(5).apply(lambda x: f"  - {x} mentions").to_string() if any(report_df['hashtags']) else '  - No hashtags found'}

---
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Save report
        filename = f"social_media_report_{topic.lower().replace(' ', '_')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"   Report saved as: {filename}")
        print(report)
        
        return report

# Example usage and main interface
def main():
    """Main interface for the Social Media Analyst Bot"""
    
    try:
        # Initialize the bot
        analyst = SimpleSocialMediaAnalyst()
        
        # Load your CSV files (adjust paths as needed)
        csv_files = [
            "social_media_posts_dakota_access_pipeline.csv",
            "social_media_posts_conocophillips_willow_project.csv", 
            "social_media_posts_texas_border_wall.csv"
        ]
        
        # Load data
        df = analyst.load_data(csv_files)
        
        print(f"\nSocial Media Intelligence Analyst Bot is ready!")
        print(f"Loaded {len(df)} posts for analysis")
        print("\n" + "="*60)
        
        # Interactive mode
        while True:
            print("\nWhat would you like to analyze?")
            print("1. Ask a question about the data")
            print("2. Generate visualizations") 
            print("3. Create analysis report")
            print("4. Show data summary")
            print("5. Sentiment analysis")
            print("6. User analysis")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                question = input("\nEnter your question: ")
                analyst.ask_question(question)
                
            elif choice == '2':
                topic = input("\nTopic for visualization (or 'overall'): ")
                analyst.create_visualizations(topic)
                
            elif choice == '3':
                topic = input("\nTopic for report (or 'overall'): ")
                analyst.generate_report(topic)
                
            elif choice == '4':
                print(f"\nData Summary:")
                print(f"Total posts: {len(df)}")
                print(f"Date range: {df['posted_at'].min()} to {df['posted_at'].max()}")
                print(f"Keywords: {', '.join(df['keyword'].unique())}")
                print(f"Sentiment distribution:\n{df['sentiment_label'].value_counts()}")
                
            elif choice == '5':
                topic = input("\nTopic for sentiment analysis (or press Enter for all): ")
                topic = topic if topic.strip() else None
                sentiment_data = analyst.get_sentiment_analysis(topic)
                print(f"\nSentiment Analysis Results:")
                if isinstance(sentiment_data, dict):
                    for key, value in sentiment_data.items():
                        print(f"{key}: {value}")
                else:
                    print(sentiment_data)
                    
            elif choice == '6':
                topic = input("\nTopic for user analysis (or press Enter for all): ")
                topic = topic if topic.strip() else None
                user_data = analyst.get_user_analysis(topic)
                print(f"\nUser Analysis Results:")
                if isinstance(user_data, dict):
                    for key, value in user_data.items():
                        print(f"{key}: {value}")
                else:
                    print(user_data)
                
            elif choice == '7':
                print("\nThank you for using Social Media Intelligence Analyst Bot!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your CSV files and API keys.")

if __name__ == "__main__":
    main()