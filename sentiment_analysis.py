"""
Sentiment Analysis Module
==========================
Analyzes market sentiment from:
- Reddit (wallstreetbets, cryptocurrency, etc.)
- Twitter (crypto influencers, traders)
- Financial News (Benzinga, MarketWatch, etc.)

Why Sentiment Matters:
- Bullish news + bullish chart = STRONG BUY 💪
- Bearish news + bullish chart = CAUTION ⚠️
- News moves markets!
"""

import praw  # Reddit API
import tweepy  # Twitter API
from newsapi import NewsApiClient  # News API
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from textblob import TextBlob  # Sentiment analysis
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURATION
# ==========================================
class SentimentConfig:
    """Sentiment analysis configuration"""
    
    # API Keys (from environment variables)
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "trading_bot:v1.0")
    
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # Reddit subreddits to monitor
    CRYPTO_SUBREDDITS = [
        'cryptocurrency',
        'CryptoMarkets', 
        'Bitcoin',
        'ethereum'
    ]
    
    STOCK_SUBREDDITS = [
        'wallstreetbets',
        'stocks',
        'investing',
        'StockMarket'
    ]
    
    # Twitter keywords
    CRYPTO_KEYWORDS = [
        'BTC', 'Bitcoin', 'ETH', 'Ethereum',
        'crypto', 'altcoin', '#cryptocurrency'
    ]
    
    STOCK_KEYWORDS = [
        'stocks', 'trading', '#stockmarket',
        '$SPY', '$QQQ', 'NYSE', 'NASDAQ'
    ]
    
    # News sources
    NEWS_SOURCES = [
        'bloomberg',
        'reuters',
        'cnbc',
        'financial-times',
        'the-wall-street-journal'
    ]
    
    # Storage
    SENTIMENT_CACHE_DIR = Path("sentiment_cache")


# ==========================================
# SENTIMENT ANALYZER
# ==========================================
class SentimentAnalyzer:
    """
    Analyzes market sentiment from multiple sources.
    
    How it works:
    1. Collects posts/tweets/news about an asset
    2. Analyzes text sentiment (-1 to +1)
    3. Aggregates into single sentiment score
    4. Combines with technical analysis
    
    Example:
        Technical: BUY signal (RSI 28, MACD bullish)
        Sentiment: Very Bullish (+0.8)
        Combined: STRONG BUY! 💪
    """
    
    def __init__(self):
        self.sentiment_cache_dir = SentimentConfig.SENTIMENT_CACHE_DIR
        self.sentiment_cache_dir.mkdir(exist_ok=True)
        
        # Initialize APIs
        self.reddit = self._init_reddit()
        self.twitter = self._init_twitter()
        self.news_api = self._init_news()
        
        logger.info("✅ Sentiment Analyzer initialized")
    
    def _init_reddit(self) -> Optional[praw.Reddit]:
        """Initialize Reddit API"""
        if not SentimentConfig.REDDIT_CLIENT_ID:
            logger.warning("⚠️ Reddit API not configured (optional)")
            return None
        
        try:
            reddit = praw.Reddit(
                client_id=SentimentConfig.REDDIT_CLIENT_ID,
                client_secret=SentimentConfig.REDDIT_CLIENT_SECRET,
                user_agent=SentimentConfig.REDDIT_USER_AGENT
            )
            logger.info("✅ Reddit API connected")
            return reddit
        except Exception as e:
            logger.error(f"❌ Reddit API failed: {e}")
            return None
    
    def _init_twitter(self) -> Optional[tweepy.Client]:
        """Initialize Twitter API"""
        if not SentimentConfig.TWITTER_BEARER_TOKEN:
            logger.warning("⚠️ Twitter API not configured (optional)")
            return None
        
        try:
            client = tweepy.Client(bearer_token=SentimentConfig.TWITTER_BEARER_TOKEN)
            logger.info("✅ Twitter API connected")
            return client
        except Exception as e:
            logger.error(f"❌ Twitter API failed: {e}")
            return None
    
    def _init_news(self) -> Optional[NewsApiClient]:
        """Initialize News API"""
        if not SentimentConfig.NEWS_API_KEY:
            logger.warning("⚠️ News API not configured (optional)")
            return None
        
        try:
            newsapi = NewsApiClient(api_key=SentimentConfig.NEWS_API_KEY)
            logger.info("✅ News API connected")
            return newsapi
        except Exception as e:
            logger.error(f"❌ News API failed: {e}")
            return None
    
    def analyze_asset_sentiment(
        self,
        asset: str,
        asset_type: str = "crypto"
    ) -> Dict:
        """
        Analyze sentiment for a specific asset.
        
        Args:
            asset: Asset symbol (e.g., 'BTC/USDT', 'AAPL')
            asset_type: 'crypto' or 'stock'
        
        Returns:
            {
                "sentiment_score": -1.0 to +1.0,
                "sentiment_label": "VERY_BEARISH" to "VERY_BULLISH",
                "confidence": 0-100,
                "sources": {
                    "reddit": {...},
                    "twitter": {...},
                    "news": {...}
                },
                "signal_boost": 0-20  # How much to boost/reduce signal confidence
            }
        """
        logger.info(f"📊 Analyzing sentiment for {asset}...")
        
        # Clean asset name for search
        search_term = self._clean_asset_name(asset, asset_type)
        
        results = {
            "asset": asset,
            "timestamp": datetime.now(),
            "sentiment_score": 0.0,
            "sentiment_label": "NEUTRAL",
            "confidence": 0,
            "sources": {},
            "signal_boost": 0
        }
        
        # Collect sentiment from each source
        sentiments = []
        
        # 1. Reddit sentiment
        if self.reddit:
            reddit_sentiment = self._analyze_reddit(search_term, asset_type)
            results["sources"]["reddit"] = reddit_sentiment
            if reddit_sentiment["post_count"] > 0:
                sentiments.append(reddit_sentiment["sentiment"])
                logger.info(f"  Reddit: {reddit_sentiment['sentiment']:.2f} ({reddit_sentiment['post_count']} posts)")
        
        # 2. Twitter sentiment
        if self.twitter:
            twitter_sentiment = self._analyze_twitter(search_term)
            results["sources"]["twitter"] = twitter_sentiment
            if twitter_sentiment["tweet_count"] > 0:
                sentiments.append(twitter_sentiment["sentiment"])
                logger.info(f"  Twitter: {twitter_sentiment['sentiment']:.2f} ({twitter_sentiment['tweet_count']} tweets)")
        
        # 3. News sentiment
        if self.news_api:
            news_sentiment = self._analyze_news(search_term)
            results["sources"]["news"] = news_sentiment
            if news_sentiment["article_count"] > 0:
                # News is more important, weight it 2x
                sentiments.append(news_sentiment["sentiment"])
                sentiments.append(news_sentiment["sentiment"])
                logger.info(f"  News: {news_sentiment['sentiment']:.2f} ({news_sentiment['article_count']} articles)")
        
        # Aggregate sentiment
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            results["sentiment_score"] = avg_sentiment
            results["confidence"] = min(100, len(sentiments) * 20)  # More sources = more confident
            
            # Label
            if avg_sentiment > 0.5:
                results["sentiment_label"] = "VERY_BULLISH"
                results["signal_boost"] = 15
            elif avg_sentiment > 0.2:
                results["sentiment_label"] = "BULLISH"
                results["signal_boost"] = 10
            elif avg_sentiment > -0.2:
                results["sentiment_label"] = "NEUTRAL"
                results["signal_boost"] = 0
            elif avg_sentiment > -0.5:
                results["sentiment_label"] = "BEARISH"
                results["signal_boost"] = -10
            else:
                results["sentiment_label"] = "VERY_BEARISH"
                results["signal_boost"] = -15
            
            logger.info(
                f"✅ Overall Sentiment: {results['sentiment_label']} "
                f"({results['sentiment_score']:.2f}, confidence: {results['confidence']}%)"
            )
        else:
            logger.warning("⚠️ No sentiment data available")
        
        return results
    
    def _clean_asset_name(self, asset: str, asset_type: str) -> str:
        """Convert asset to searchable term"""
        if asset_type == "crypto":
            # BTC/USDT → Bitcoin
            mappings = {
                'BTC': 'Bitcoin',
                'ETH': 'Ethereum',
                'BNB': 'Binance Coin',
                'SOL': 'Solana',
                'ADA': 'Cardano',
                'AVAX': 'Avalanche',
                'MATIC': 'Polygon',
                'DOT': 'Polkadot',
                'LINK': 'Chainlink'
            }
            base = asset.split('/')[0]
            return mappings.get(base, base)
        else:
            # Stock ticker as-is
            return asset
    
    def _analyze_reddit(self, search_term: str, asset_type: str) -> Dict:
        """Analyze Reddit sentiment"""
        if not self.reddit:
            return {"sentiment": 0, "post_count": 0}
        
        try:
            subreddits = (SentimentConfig.CRYPTO_SUBREDDITS 
                         if asset_type == "crypto" 
                         else SentimentConfig.STOCK_SUBREDDITS)
            
            posts = []
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    # Get recent posts
                    for post in subreddit.search(search_term, time_filter='day', limit=10):
                        posts.append({
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'timestamp': datetime.fromtimestamp(post.created_utc)
                        })
                except Exception as e:
                    logger.debug(f"Error in r/{subreddit_name}: {e}")
            
            if not posts:
                return {"sentiment": 0, "post_count": 0}
            
            # Analyze sentiment
            sentiments = []
            for post in posts:
                text = f"{post['title']} {post['text']}"
                blob = TextBlob(text)
                # Weight by post score (upvotes)
                weight = max(1, post['score'])
                sentiments.extend([blob.sentiment.polarity] * weight)
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                "sentiment": avg_sentiment,
                "post_count": len(posts),
                "avg_score": sum(p['score'] for p in posts) / len(posts)
            }
            
        except Exception as e:
            logger.error(f"Reddit analysis failed: {e}")
            return {"sentiment": 0, "post_count": 0}
    
    def _analyze_twitter(self, search_term: str) -> Dict:
        """Analyze Twitter sentiment"""
        if not self.twitter:
            return {"sentiment": 0, "tweet_count": 0}
        
        try:
            # Search recent tweets
            tweets = self.twitter.search_recent_tweets(
                query=f"{search_term} -is:retweet lang:en",
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return {"sentiment": 0, "tweet_count": 0}
            
            # Analyze sentiment
            sentiments = []
            for tweet in tweets.data:
                blob = TextBlob(tweet.text)
                # Weight by likes + retweets
                metrics = tweet.public_metrics
                weight = max(1, metrics['like_count'] + metrics['retweet_count'])
                sentiments.extend([blob.sentiment.polarity] * weight)
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                "sentiment": avg_sentiment,
                "tweet_count": len(tweets.data)
            }
            
        except Exception as e:
            logger.error(f"Twitter analysis failed: {e}")
            return {"sentiment": 0, "tweet_count": 0}
    
    def _analyze_news(self, search_term: str) -> Dict:
        """Analyze news sentiment"""
        if not self.news_api:
            return {"sentiment": 0, "article_count": 0}
        
        try:
            # Get recent news
            news = self.news_api.get_everything(
                q=search_term,
                sources=','.join(SentimentConfig.NEWS_SOURCES),
                from_param=(datetime.now() - timedelta(days=1)).isoformat(),
                language='en',
                sort_by='relevancy'
            )
            
            articles = news.get('articles', [])
            if not articles:
                return {"sentiment": 0, "article_count": 0}
            
            # Analyze sentiment
            sentiments = []
            for article in articles:
                text = f"{article['title']} {article.get('description', '')}"
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                "sentiment": avg_sentiment,
                "article_count": len(articles),
                "top_headline": articles[0]['title'] if articles else None
            }
            
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return {"sentiment": 0, "article_count": 0}


# ==========================================
# COMBINED SIGNAL GENERATOR
# ==========================================
def combine_technical_and_sentiment(
    technical_analysis: Dict,
    sentiment_analysis: Dict
) -> Dict:
    """
    Combine technical signals with sentiment.
    
    Rules:
    - Bullish chart + bullish news = STRONGER BUY
    - Bullish chart + bearish news = CAUTION
    - Bearish chart + bearish news = STRONGER SELL
    - Bearish chart + bullish news = WAIT
    
    Args:
        technical_analysis: From strategy_engine.analyze_market()
        sentiment_analysis: From SentimentAnalyzer.analyze_asset_sentiment()
    
    Returns:
        Enhanced analysis with sentiment-adjusted confidence
    """
    # Get original values
    signal = technical_analysis.get('signal', 'HOLD')
    original_confidence = technical_analysis.get('confidence', 0)
    sentiment_boost = sentiment_analysis.get('signal_boost', 0)
    
    # Adjust confidence based on sentiment
    if signal == "BUY":
        # If sentiment is also bullish, boost confidence
        if sentiment_boost > 0:
            adjusted_confidence = min(100, original_confidence + sentiment_boost)
            reasoning = f"✅ Sentiment confirms: {sentiment_analysis['sentiment_label']}"
        else:
            # Sentiment is bearish, reduce confidence
            adjusted_confidence = max(0, original_confidence + sentiment_boost)
            reasoning = f"⚠️ Sentiment disagrees: {sentiment_analysis['sentiment_label']}"
    
    elif signal == "SELL":
        # If sentiment is also bearish, boost confidence
        if sentiment_boost < 0:
            adjusted_confidence = min(100, original_confidence + abs(sentiment_boost))
            reasoning = f"✅ Sentiment confirms: {sentiment_analysis['sentiment_label']}"
        else:
            # Sentiment is bullish, reduce confidence
            adjusted_confidence = max(0, original_confidence - sentiment_boost)
            reasoning = f"⚠️ Sentiment disagrees: {sentiment_analysis['sentiment_label']}"
    
    else:  # HOLD
        adjusted_confidence = original_confidence
        reasoning = "No sentiment adjustment for HOLD signal"
    
    # Create enhanced result
    result = technical_analysis.copy()
    result['original_confidence'] = original_confidence
    result['sentiment_adjusted_confidence'] = adjusted_confidence
    result['sentiment_analysis'] = sentiment_analysis
    result['sentiment_reasoning'] = reasoning
    result['confidence'] = adjusted_confidence  # Update main confidence
    
    return result


# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    """
    Test sentiment analysis
    
    Setup first:
    1. Get API keys:
       - Reddit: https://www.reddit.com/prefs/apps
       - Twitter: https://developer.twitter.com/
       - News: https://newsapi.org/
    
    2. Set environment variables:
       export REDDIT_CLIENT_ID=your_id
       export REDDIT_CLIENT_SECRET=your_secret
       export TWITTER_BEARER_TOKEN=your_token
       export NEWS_API_KEY=your_key
    """
    
    print("\n" + "="*60)
    print("📊 SENTIMENT ANALYSIS TEST")
    print("="*60)
    
    analyzer = SentimentAnalyzer()
    
    # Test Bitcoin sentiment
    print("\n💰 Analyzing Bitcoin sentiment...")
    btc_sentiment = analyzer.analyze_asset_sentiment("BTC/USDT", "crypto")
    
    print(f"\nResult:")
    print(f"  Sentiment: {btc_sentiment['sentiment_label']}")
    print(f"  Score: {btc_sentiment['sentiment_score']:.2f}")
    print(f"  Confidence: {btc_sentiment['confidence']}%")
    print(f"  Signal Boost: {btc_sentiment['signal_boost']:+d}%")
    
    # Test combining with technical
    print("\n🔧 Testing combined signal...")
    fake_technical = {
        "signal": "BUY",
        "confidence": 70,
        "reasoning": ["RSI oversold", "MACD bullish"]
    }
    
    combined = combine_technical_and_sentiment(fake_technical, btc_sentiment)
    
    print(f"\nCombined Result:")
    print(f"  Original Confidence: {fake_technical['confidence']}%")
    print(f"  Sentiment Adjustment: {btc_sentiment['signal_boost']:+d}%")
    print(f"  Final Confidence: {combined['confidence']}%")
    print(f"  Reasoning: {combined['sentiment_reasoning']}")
    
    print("\n" + "="*60)