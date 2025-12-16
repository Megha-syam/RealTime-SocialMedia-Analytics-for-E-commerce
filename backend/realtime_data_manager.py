"""
Real-time Data Manager for Real-Time Social Media Analytics for E-Commerce Trends
Handles continuous data fetching and processing without database storage
"""

import asyncio
import httpx
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, asdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    timestamp: datetime
    sku: str
    sentiment_score: float
    news_volume: int
    economic_impact: float
    confidence: float

@dataclass
class NewsArticle:
    id: str
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    sentiment_label: str
    skus_mentioned: List[str]

@dataclass
class EconomicEvent:
    event: str
    country: str
    date: str
    actual: str
    forecast: str
    previous: str
    importance: str
    sentiment_score: float
    market_impact: float

class RealTimeDataManager:
    def __init__(self):
        self.news_api_key = "pub_f6cb8108c3dd449299f965cd7c131702"
        self.economic_api_key = "b4a2da19-2dec-4a25-a97e-290d452a91d4"
        self.gemini_api_key = "AIzaSyCnvqMspISmF_IAFkTO89czQkRxoKHLQzY"

        # In-memory storage for real-time data
        self.news_data: List[NewsArticle] = []
        self.economic_data: List[EconomicEvent] = []
        self.market_data: Dict[str, List[MarketDataPoint]] = {}
        self.last_update = None

        # Threading lock to protect shared structures when using sync threads
        self._data_lock = threading.Lock()

        # Popular symbols to track
        # Track popular product SKUs (example SKUs; update to your catalog)
        self.tracked_skus = [
            "SKU-AAPL-01", "SKU-GOOGL-01", "SKU-MSFT-01", "SKU-AMZN-01", "SKU-TSLA-01",
            "SKU-NVDA-01", "SKU-META-01", "SKU-NFLX-01", "SKU-AMD-01", "SKU-INTC-01"
        ]

        # Reddit ingestion configuration (use environment variables)
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "sentiment-dashboard-reddit")
        self.reddit_subreddits = os.getenv("REDDIT_SUBREDDITS", "ecommerce,shopping,deals,amazon,flipkart").split(",")
        self.reddit_enabled = bool(self.reddit_client_id and self.reddit_client_secret)
        # Control whether to fetch upstream news APIs (set to 'true' to enable). Default: false (Reddit-only)
        self.use_upstream_news = os.getenv("USE_UPSTREAM_NEWS", "false").lower() == "true"
        # Keywords to filter purchase-related posts
        self.purchase_keywords = ["bought", "purchased", "ordered", "deal", "price", "sale", "discount"]
        
    async def start_realtime_monitoring(self):
        """Start continuous real-time data monitoring"""
        logger.info("Starting real-time market sentiment monitoring...")
        # Start Reddit ingestion thread if configured
        if self.reddit_enabled:
            t = threading.Thread(target=self._reddit_ingest_worker, daemon=True)
            t.start()
        else:
            logger.info("Reddit ingestion not configured (set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to enable)")
        # Create tasks for different data sources. Upstream news optional.
        tasks = []
        if self.use_upstream_news:
            tasks.append(asyncio.create_task(self.continuous_news_monitoring()))
        tasks.extend([
            asyncio.create_task(self.continuous_economic_monitoring()),
            asyncio.create_task(self.continuous_sentiment_calculation()),
            asyncio.create_task(self.data_cleanup_task())
        ])
        
        await asyncio.gather(*tasks)
    
    async def continuous_news_monitoring(self):
        """Continuously monitor news sources"""
        while True:
            try:
                await self.fetch_latest_news()
                logger.info(f"News data updated: {len(self.news_data)} articles")
                await asyncio.sleep(900)  # 15 minutes
            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")
                await asyncio.sleep(300)  # 5 minutes retry
    
    async def continuous_economic_monitoring(self):
        """Continuously monitor economic events"""
        while True:
            try:
                await self.fetch_economic_events()
                logger.info(f"Economic data updated: {len(self.economic_data)} events")
                await asyncio.sleep(1800)  # 30 minutes
            except Exception as e:
                logger.error(f"Error in economic monitoring: {e}")
                await asyncio.sleep(600)  # 10 minutes retry
    
    async def continuous_sentiment_calculation(self):
        """Continuously calculate sentiment for tracked symbols"""
        while True:
            try:
                for sku in self.tracked_skus:
                    await self.calculate_sku_sentiment(sku)

                logger.info(f"Sentiment calculated for {len(self.tracked_skus)} SKUs")
                await asyncio.sleep(600)  # 10 minutes
            except Exception as e:
                logger.error(f"Error in sentiment calculation: {e}")
                await asyncio.sleep(300)  # 5 minutes retry
    
    async def fetch_latest_news(self):
        """Fetch latest news from multiple sources"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Fetch general business news
                url = "https://newsdata.io/api/1/news"
                params = {
                    "apikey": self.news_api_key,
                    "category": "business",
                    "language": "en",
                    "country": "us",
                    "size": 50
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Process articles
                new_articles = []
                for article in data.get("results", []):
                    if article.get("title") and article.get("description"):
                        processed_article = await self.process_news_article(article)
                        if processed_article:
                            new_articles.append(processed_article)
                
                # Update news data (keep only last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                with self._data_lock:
                    self.news_data = [
                        article for article in self.news_data 
                        if article.published_at > cutoff_time
                    ] + new_articles
                    self.last_update = datetime.now()
                
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
    
    async def process_news_article(self, article: dict) -> Optional[NewsArticle]:
        """Process and analyze a single news article"""
        try:
            # Extract text for sentiment analysis
            text = f"{article['title']} {article.get('description', '')}"
            
            # Analyze sentiment
            sentiment_score, sentiment_label = self.analyze_text_sentiment(text)
            
            # Detect mentioned SKUs/products
            skus_mentioned = self.extract_skus_from_text(text)
            
            return NewsArticle(
                id=article.get("article_id", ""),
                title=article["title"],
                description=article.get("description", ""),
                source=article.get("source_id", "Unknown"),
                url=article.get("link", ""),
                published_at=datetime.fromisoformat(
                    article.get("pubDate", datetime.now().isoformat()).replace('Z', '+00:00')
                ),
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                skus_mentioned=skus_mentioned
            )
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    async def fetch_economic_events(self):
        """Fetch latest economic events"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                url = f"https://api.tradingeconomics.com/calendar?c={self.economic_api_key}"
                
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Process events
                new_events = []
                for event in data[:30]:  # Limit to recent events
                    if event.get("Event") and event.get("Country"):
                        processed_event = self.process_economic_event(event)
                        if processed_event:
                            new_events.append(processed_event)
                
                # Update economic data
                self.economic_data = new_events
                
            except Exception as e:
                logger.error(f"Error fetching economic events: {e}")
    
    def process_economic_event(self, event: dict) -> Optional[EconomicEvent]:
        """Process a single economic event"""
        try:
            # Analyze impact sentiment
            impact_text = f"{event.get('Event', '')} {event.get('Country', '')}"
            sentiment_score, _ = self.analyze_text_sentiment(impact_text)
            
            # Calculate market impact based on importance
            importance = event.get("Importance", "Low")
            market_impact = {
                "High": 0.8,
                "Medium": 0.5,
                "Low": 0.2
            }.get(importance, 0.2)
            
            return EconomicEvent(
                event=event.get("Event", ""),
                country=event.get("Country", ""),
                date=event.get("Date", ""),
                actual=str(event.get("Actual", "")),
                forecast=str(event.get("Forecast", "")),
                previous=str(event.get("Previous", "")),
                importance=importance,
                sentiment_score=sentiment_score,
                market_impact=market_impact
            )
        except Exception as e:
            logger.error(f"Error processing economic event: {e}")
            return None
    
    async def calculate_symbol_sentiment(self, symbol: str):
        """Calculate real-time sentiment for a specific SKU"""
        try:
            sku = symbol  # keep the public param name `symbol` for compatibility
            # Get relevant news articles
            relevant_news = [
                article for article in self.news_data
                if sku in article.skus_mentioned or 
                   sku.lower() in article.title.lower() or
                   sku.lower() in article.description.lower()
            ]
            
            # Calculate news sentiment
            if relevant_news:
                news_sentiment = sum(article.sentiment_score for article in relevant_news) / len(relevant_news)
                news_volume = len(relevant_news)
            else:
                news_sentiment = 0.0
                news_volume = 0
            
            # Calculate economic sentiment impact
            economic_sentiment = sum(event.sentiment_score * event.market_impact for event in self.economic_data)
            economic_sentiment = economic_sentiment / len(self.economic_data) if self.economic_data else 0.0
            
            # Weighted overall sentiment
            overall_sentiment = (
                news_sentiment * 0.6 +
                economic_sentiment * 0.4
            )
            
            # Calculate confidence based on data volume
            confidence = min(0.95, max(0.3, (news_volume / 10) + 0.3))
            
            # Create market data point
            data_point = MarketDataPoint(
                timestamp=datetime.now(),
                sku=sku,
                sentiment_score=overall_sentiment,
                news_volume=news_volume,
                economic_impact=economic_sentiment,
                confidence=confidence
            )
            
            # Store in market data
            if sku not in self.market_data:
                self.market_data[sku] = []

            self.market_data[sku].append(data_point)
            
            # Keep only last 24 hours of data
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.market_data[sku] = [
                point for point in self.market_data[sku]
                if point.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {e}")

    # Backwards-compatible alias: some callers expect calculate_sku_sentiment
    async def calculate_sku_sentiment(self, symbol: str):
        return await self.calculate_symbol_sentiment(symbol)
    
    def analyze_text_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment of text using e-commerce/product keywords"""
        positive_words = [
            'bestseller', 'popular', 'positive review', 'high rating', 'love', 'recommend',
            'fast sell', 'in demand', 'restock', 'promotion', 'discount_success'
        ]
        negative_words = [
            'return', 'poor review', 'low rating', 'complaint', 'recall', 'out of stock',
            'delay', 'broken', 'refund', 'defect'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score
        if positive_count > negative_count:
            score = min(0.9, positive_count * 0.15)
            label = "positive"
        elif negative_count > positive_count:
            score = max(-0.9, -negative_count * 0.15)
            label = "negative"
        else:
            score = 0.0
            label = "neutral"
        
        return score, label
    
    def extract_symbols_from_text(self, text: str) -> List[str]:
        """Legacy: Extract SKUs mentioned in text (keeps old name for compatibility)
        Use `extract_skus_from_text` instead."""
        return self.extract_skus_from_text(text)

    def extract_skus_from_text(self, text: str) -> List[str]:
        """Extract product SKUs mentioned in text"""
        skus_found: List[str] = []
        text_upper = text.upper()

        for sku in self.tracked_skus:
            if sku.upper() in text_upper:
                skus_found.append(sku)

        return skus_found

    # -------------------------
    # Reddit ingestion (synchronous worker running in a background thread)
    # -------------------------
    def _reddit_ingest_worker(self):
        """Background worker that uses PRAW to ingest Reddit submissions.

        This runs in a separate thread to avoid blocking the asyncio event loop.
        It is safe to run even if PRAW is not installed (it will log a warning).
        """
        try:
            import praw
        except Exception:
            logger.warning("praw not available; Reddit ingestion disabled")
            return

        if not self.reddit_enabled:
            logger.info("Reddit ingestion is not enabled (missing credentials)")
            return

        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent,
            )

            logger.info(f"Starting Reddit ingestion for: {', '.join(self.reddit_subreddits)}")

            # Use subreddit streams; skip_existing avoids processing backlog on first run
            for sub in self.reddit_subreddits:
                try:
                    subreddit = reddit.subreddit(sub)
                except Exception as e:
                    logger.error(f"Error accessing subreddit {sub}: {e}")
                    continue

                # Iterate submissions in a non-blocking manner by calling .new() repeatedly
                while True:
                    try:
                        for submission in subreddit.new(limit=50):
                            try:
                                title = getattr(submission, 'title', '') or ''
                                selftext = getattr(submission, 'selftext', '') or ''
                                content = f"{title} {selftext}".lower()

                                if not any(k in content for k in self.purchase_keywords):
                                    continue

                                # Build a NewsArticle-like object synchronously
                                text = f"{title} {selftext}"
                                sentiment_score, sentiment_label = self.analyze_text_sentiment(text)
                                skus_mentioned = self.extract_skus_from_text(text)

                                article = NewsArticle(
                                    id=f"reddit_{getattr(submission,'id','')}",
                                    title=title,
                                    description=selftext,
                                    source=f"reddit/{getattr(submission,'subreddit','')}",
                                    url=getattr(submission,'url',''),
                                    published_at=datetime.fromtimestamp(getattr(submission,'created_utc', time.time())),
                                    sentiment_score=sentiment_score,
                                    sentiment_label=sentiment_label,
                                    skus_mentioned=skus_mentioned
                                )

                                # Append to news_data with lock protection and prune old entries
                                cutoff_time = datetime.now() - timedelta(hours=24)
                                with self._data_lock:
                                    self.news_data = [
                                        a for a in self.news_data if a.published_at > cutoff_time
                                    ] + [article]
                                    self.last_update = datetime.now()

                            except Exception as e:
                                logger.debug(f"Error processing reddit submission: {e}")

                        # Sleep briefly before fetching the next batch
                        time.sleep(15)

                    except Exception as e:
                        logger.error(f"Error in reddit ingestion loop for {sub}: {e}")
                        time.sleep(60)

        except Exception as e:
            logger.error(f"Reddit ingestion failed to start: {e}")
    
    async def data_cleanup_task(self):
        """Periodic cleanup of old data"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                # Clean news data
                self.news_data = [
                    article for article in self.news_data
                    if article.published_at > cutoff_time
                ]
                
                # Clean market data
                for sku in list(self.market_data.keys()):
                    self.market_data[sku] = [
                        point for point in self.market_data[sku]
                        if point.timestamp > cutoff_time
                    ]
                
                logger.info("Data cleanup completed")
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
                await asyncio.sleep(1800)  # 30 minutes retry
    
    def get_current_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get current sentiment data for a symbol"""
        sku = symbol
        if sku not in self.market_data or not self.market_data[sku]:
            return None

        latest_point = self.market_data[sku][-1]
        return asdict(latest_point)
    
    def get_sentiment_trend(self, symbol: str, hours: int = 6) -> Dict:
        """Get sentiment trend for a symbol over specified hours"""
        sku = symbol
        if sku not in self.market_data:
            return {"trend": "neutral", "change": 0.0, "data_points": 0}

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            point for point in self.market_data[sku]
            if point.timestamp > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return {"trend": "neutral", "change": 0.0, "data_points": len(recent_data)}
        
        # Calculate trend
        current_sentiment = recent_data[-1].sentiment_score
        previous_sentiment = recent_data[0].sentiment_score
        change = current_sentiment - previous_sentiment
        
        if change > 0.1:
            trend = "bullish"
        elif change < -0.1:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "trend": trend,
            "change": change,
            "data_points": len(recent_data),
            "current_sentiment": current_sentiment,
            "previous_sentiment": previous_sentiment
        }
    
    def get_market_overview(self) -> Dict:
        """Get overall market sentiment overview"""
        if not self.news_data:
            return {"overall_sentiment": 50, "market_mood": "neutral"}
        
        # Calculate average sentiment from all news
        total_sentiment = sum(article.sentiment_score for article in self.news_data)
        avg_sentiment = total_sentiment / len(self.news_data)
        
        # Convert to 0-100 scale
        overall_sentiment = max(0, min(100, (avg_sentiment + 1) * 50))
        
        # Determine market mood
        if overall_sentiment > 65:
            mood = "very_bullish"
        elif overall_sentiment > 55:
            mood = "bullish"
        elif overall_sentiment < 35:
            mood = "very_bearish"
        elif overall_sentiment < 45:
            mood = "bearish"
        else:
            mood = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "market_mood": mood,
            "news_volume": len(self.news_data),
            "economic_events": len(self.economic_data),
            "tracked_symbols": len(self.tracked_skus),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }

    def get_live_trending_keywords(self, symbol: str = None, top_n: int = 10) -> Dict:
        """Compute trending e-commerce keywords from in-memory news_data (no DB required).

        If `symbol` is provided, filter articles that mention the symbol (SKU) or the SKU appears in skus_mentioned.
        Returns top_n keywords and counts and a simple sentiment breakdown.
        """
        from collections import Counter
        import re

        with self._data_lock:
            articles = list(self.news_data)

        if symbol:
            s_up = symbol.upper()
            filtered = [a for a in articles if s_up in (a.skus_mentioned or []) or s_up in (a.title or '').upper() or s_up in (a.description or '').upper()]
        else:
            filtered = articles

        # Collect text
        texts = []
        for a in filtered:
            texts.append(' '.join(filter(None, [a.title or '', a.description or ''])))

        # Simple tokenization
        pattern = re.compile(r"\b[\w'&-]{3,}\b")
        stopwords = set(['the','and','for','with','that','this','from','your','you','are','was','were','have','has','had','product','item','buy','purchase','order','sale','price','discount','deal'])

        words = []
        for t in texts:
            t_low = (t or '').lower()
            for m in pattern.findall(t_low):
                if m in stopwords:
                    continue
                if m.isnumeric():
                    continue
                words.append(m)

        counts = Counter(words)
        top = counts.most_common(top_n)

        # sentiment breakdown
        pos = neg = neu = 0
        for a in filtered:
            if a.sentiment_label:
                sl = (a.sentiment_label or '').lower()
                if 'positive' in sl:
                    pos += 1
                elif 'negative' in sl:
                    neg += 1
                else:
                    neu += 1

        total = pos + neg + neu if (pos + neg + neu) > 0 else 1

        return {
            'for_symbol': symbol,
            'total_articles': len(filtered),
            'keywords': [k for k,_ in top],
            'counts': {k: int(v) for k,v in top},
            'sentiment': {
                'positive': pos,
                'negative': neg,
                'neutral': neu,
                'positive_pct': round(pos/total*100,1),
                'negative_pct': round(neg/total*100,1),
                'neutral_pct': round(neu/total*100,1)
            }
        }

# Global instance
data_manager = RealTimeDataManager()

async def start_data_monitoring():
    """Start the real-time data monitoring system"""
    await data_manager.start_realtime_monitoring()

if __name__ == "__main__":
    asyncio.run(start_data_monitoring())
