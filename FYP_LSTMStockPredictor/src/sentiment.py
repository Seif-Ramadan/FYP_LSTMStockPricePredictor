import praw
import time
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_reddit_instance():
    """
    Create and return a PRAW Reddit instance.
    """
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID", 
        client_secret="YOUR_CLIENT_SECRET",   
        user_agent="StockSentimentApp/0.0.1 by u/YourRedditUsername"
    )
    return reddit

def fetch_reddit_comments_about_ticker(ticker, subreddits=None, min_upvotes=3, days_back=3):
    """
    Fetch comments from the last X days containing the ticker symbol from specified subreddits.
    """
    if subreddits is None:
        # Add more popular subreddits for finance/stocks
        subreddits = ["Stocks", "Investing", "wallstreetbets", "StockMarket"]

    reddit = get_reddit_instance()
    
    results = []
    time_now = time.time()
    days_ago = time_now - (days_back * 24 * 60 * 60)

    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)

            # We'll search by ticker over the last month for more results
            for submission in subreddit.search(
                query=ticker,
                sort="new",
                time_filter="month",
                limit=200
            ):
                # Retrieve comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    # Filter by age
                    if comment.created_utc >= days_ago:
                        text_lower = comment.body.lower()
                        # Check if comment explicitly references the ticker
                        if ticker.lower() in text_lower or f"${ticker.lower()}" in text_lower:
                            # Check upvotes & basic length
                            if comment.score >= min_upvotes and len(text_lower) > 20:
                                results.append({
                                    "text": comment.body,
                                    "score": comment.score,
                                    "created_utc": comment.created_utc,
                                    "subreddit": subreddit_name
                                })

        except Exception as ex:
            # If we get an API error or forbidden subreddit, just skip
            print(f"Skipping subreddit {subreddit_name} due to error: {ex}")
            continue

    return results

def analyze_sentiment_ratio(comments):
    
    sid = SentimentIntensityAnalyzer()
    pos_count, neg_count, neu_count = 0, 0, 0
    total_sentiment = 0
    weighted_sentiment = 0
    total_weight = 0

    for c in comments:
        text = c['text']
        scores = sid.polarity_scores(text)
        
        # Use comment score as weight
        weight = c['score']
        total_weight += weight
        weighted_sentiment += scores['compound'] * weight
        total_sentiment += scores['compound']
        
        if scores['compound'] > 0.05:
            pos_count += 1
        elif scores['compound'] < -0.05:
            neg_count += 1
        else:
            neu_count += 1

    total = len(comments)
    ratio = round(pos_count / neg_count, 2) if neg_count else float('inf')
    avg_sentiment = total_sentiment / total if total else 0
    weighted_sentiment_avg = weighted_sentiment / total_weight if total_weight else 0

    return {
        'total_comments': total,
        'positive_comments': pos_count,
        'negative_comments': neg_count,
        'neutral_comments': neu_count,
        'ratio_pos_neg': ratio,
        'average_sentiment': avg_sentiment,
        'weighted_sentiment': weighted_sentiment_avg
    }

def get_sentiment_by_subreddit(comments):
    """
    Break down sentiment by subreddit.
    """
    sid = SentimentIntensityAnalyzer()
    subreddit_data = {}
    
    for c in comments:
        subreddit = c['subreddit']
        text = c['text']
        scores = sid.polarity_scores(text)
        
        if subreddit not in subreddit_data:
            subreddit_data[subreddit] = {
                'comments': 0,
                'total_sentiment': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        
        subreddit_data[subreddit]['comments'] += 1
        subreddit_data[subreddit]['total_sentiment'] += scores['compound']
        
        if scores['compound'] > 0.05:
            subreddit_data[subreddit]['positive'] += 1
        elif scores['compound'] < -0.05:
            subreddit_data[subreddit]['negative'] += 1
        else:
            subreddit_data[subreddit]['neutral'] += 1
    
    # Calculate averages
    for sub in subreddit_data:
        count = subreddit_data[sub]['comments']
        subreddit_data[sub]['average_sentiment'] = subreddit_data[sub]['total_sentiment'] / count if count else 0
    
    return subreddit_data