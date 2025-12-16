"""
Database service layer for Real-Time Social Media Analytics for E-Commerce Trends
Handles all database operations and business logic
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import uuid

from .database import (
    Product, NewsArticle, EconomicEvent, SentimentSnapshot, 
    AIRecommendation, MarketIndicator, DataRefreshLog, SocialMention,
    SentimentLabel, RecommendationAction, RiskLevel,
    get_db, get_or_create_product, get_product_by_sku, save_sentiment_snapshot,
    save_news_article, save_ai_recommendation, log_data_refresh
)

class DatabaseService:
    def __init__(self):
        pass
    
    def get_product_sentiment_history(self, db: Session, sku: str, hours: int = 24) -> List[Dict]:
        """Get sentiment history for a product (by SKU)"""
        product = db.query(Product).filter(Product.sku == sku.upper()).first()
        if not product:
            return []

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        snapshots = db.query(SentimentSnapshot).filter(
            and_(
                SentimentSnapshot.product_id == product.id,
                SentimentSnapshot.snapshot_time >= cutoff_time
            )
        ).order_by(SentimentSnapshot.snapshot_time.asc()).all()
        
        return [
            {
                'time': snapshot.snapshot_time.strftime('%H:%M'),
                'sentiment': float(snapshot.overall_sentiment or 0),
                'volume': (snapshot.news_volume or 0) + (snapshot.social_volume or 0),
                'news_sentiment': float(snapshot.news_sentiment or 0),
                'social_sentiment': float(snapshot.social_sentiment or 0),
                'economic_sentiment': float(snapshot.economic_sentiment or 0)
            }
            for snapshot in snapshots
        ]
    
    def get_latest_news_with_sentiment(self, db: Session, limit: int = 20) -> List[Dict]:
        """Get latest news articles with sentiment analysis"""
        articles = db.query(NewsArticle).filter(
            NewsArticle.published_at >= datetime.utcnow() - timedelta(hours=24)
        ).order_by(desc(NewsArticle.published_at)).limit(limit).all()
        
        return [
            {
                'id': str(article.id),
                'title': article.title,
                'description': article.description,
                'source': article.source,
                'url': article.url,
                'published_at': article.published_at,
                'sentiment_score': float(article.sentiment_score or 0),
                'sentiment_label': article.sentiment_label.value if article.sentiment_label else 'neutral',
                'confidence': float(article.confidence or 0),
                'time': self._time_ago(article.published_at) if article.published_at else 'Unknown'
            }
            for article in articles
        ]
    
    def get_active_recommendations(self, db: Session, symbol: str = None) -> List[Dict]:
        """Get active AI recommendations"""
        query = db.query(AIRecommendation).filter(AIRecommendation.is_active == True)

        if symbol:
            product = db.query(Product).filter(Product.sku == symbol.upper()).first()
            if product:
                query = query.filter(AIRecommendation.product_id == product.id)
        
        recommendations = query.order_by(desc(AIRecommendation.generated_at)).all()
        
        result = []
        for rec in recommendations:
            product = db.query(Product).filter(Product.id == rec.product_id).first()
            result.append({
                'sku': product.sku if product else 'Unknown',
                'action': rec.recommendation_action.value,
                'confidence': rec.confidence,
                'reasoning': rec.reasoning,
                'risk_level': rec.risk_level.value if rec.risk_level else 'Medium',
                'time_horizon': rec.time_horizon,
                'generated_at': rec.generated_at
            })
        
        return result
    
    def get_market_overview(self, db: Session) -> Dict:
        """Get overall market sentiment overview"""
        # Get latest sentiment snapshots for all stocks
        latest_snapshots = db.query(
            Product.sku,
            SentimentSnapshot.overall_sentiment,
            SentimentSnapshot.news_volume,
            SentimentSnapshot.social_volume
        ).join(
            SentimentSnapshot, Product.id == SentimentSnapshot.product_id
        ).filter(
            SentimentSnapshot.snapshot_time >= datetime.utcnow() - timedelta(hours=2)
        ).order_by(
            Product.sku, desc(SentimentSnapshot.snapshot_time)
        ).distinct(Product.sku).all()
        
        if not latest_snapshots:
            return {
                'overall_sentiment': 50.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'total_volume': 0,
                'market_mood': 'Neutral'
            }
        
        sentiments = [float(s.overall_sentiment or 50) for s in latest_snapshots]
        overall_sentiment = sum(sentiments) / len(sentiments)
        
        bullish_count = sum(1 for s in sentiments if s >= 60)
        bearish_count = sum(1 for s in sentiments if s <= 40)
        neutral_count = len(sentiments) - bullish_count - bearish_count
        
        total_volume = sum((s.news_volume or 0) + (s.social_volume or 0) for s in latest_snapshots)
        
        if overall_sentiment >= 60:
            market_mood = 'Bullish'
        elif overall_sentiment <= 40:
            market_mood = 'Bearish'
        else:
            market_mood = 'Neutral'
        
        return {
            'overall_sentiment': round(overall_sentiment, 1),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_volume': total_volume,
            'market_mood': market_mood
        }
    
    def get_economic_events(self, db: Session, days: int = 7) -> List[Dict]:
        """Get upcoming economic events"""
        cutoff_date = datetime.utcnow() - timedelta(days=1)
        future_date = datetime.utcnow() + timedelta(days=days)
        
        events = db.query(EconomicEvent).filter(
            and_(
                EconomicEvent.event_date >= cutoff_date,
                EconomicEvent.event_date <= future_date
            )
        ).order_by(EconomicEvent.event_date.asc()).all()
        
        return [
            {
                'event_name': event.event_name,
                'country': event.country,
                'event_date': event.event_date,
                'actual_value': event.actual_value,
                'forecast_value': event.forecast_value,
                'previous_value': event.previous_value,
                'importance': event.importance,
                'sentiment_score': float(event.sentiment_score or 0),
                'sentiment_label': event.sentiment_label.value if event.sentiment_label else 'neutral'
            }
            for event in events
        ]

    def get_trending_keywords_for_sku(self, db: Session, sku: str, top_n: int = 10) -> Dict:
        """Compute trending e-commerce keywords for a SKU/product using recent news, social mentions, and recommendations.

        Returns a dict with top keywords and simple sentiment breakdown.
        """
        from collections import Counter
        import re
        # basic stopwords list (small to avoid heavy deps)
        stopwords = set([
            'the','and','for','with','that','this','from','your','you','are','was','were','have','has','had',
            'product','item','buy','purchase','order','orders','sale','price','discount','deal'
        ])

        product = db.query(Product).filter(Product.sku == sku.upper()).first()
        if not product:
            return {'sku': sku.upper(), 'keywords': [], 'counts': {}, 'sentiment_summary': {}}

        cutoff = datetime.utcnow() - timedelta(days=7)

        # Gather text from news articles
        news_texts = db.query(NewsArticle).filter(
            NewsArticle.published_at >= cutoff
        ).order_by(desc(NewsArticle.published_at)).limit(200).all()

        collected = []
        for a in news_texts:
            text = ' '.join(filter(None, [a.title or '', a.description or '', a.content or '']))
            # only include if mentions sku or product name
            if product.sku.upper() in (text or '').upper() or (product.name and product.name.lower() in (text or '').lower()):
                collected.append(text)

        # Gather social mentions for product
        social_rows = db.query(SocialMention).filter(
            SocialMention.product_id == product.id,
            SocialMention.posted_at >= cutoff
        ).order_by(desc(SocialMention.posted_at)).limit(500).all()

        for s in social_rows:
            if s.content:
                collected.append(s.content)

        # Gather recent AI recommendation reasoning text
        recs = db.query(AIRecommendation).filter(
            AIRecommendation.product_id == product.id,
            AIRecommendation.generated_at >= cutoff
        ).order_by(desc(AIRecommendation.generated_at)).limit(50).all()
        for r in recs:
            if r.reasoning:
                collected.append(r.reasoning)

        # Tokenize and count keywords
        words = []
        pattern = re.compile(r"\b[\w'&-]{3,}\b")
        for text in collected:
            text_lower = (text or '').lower()
            for m in pattern.findall(text_lower):
                if m in stopwords:
                    continue
                # strip numeric-only tokens
                if m.isnumeric():
                    continue
                words.append(m)

        counts = Counter(words)
        top = counts.most_common(top_n)

        # Simple sentiment summary from news and social
        pos = neg = neu = 0
        for a in news_texts:
            if a.sentiment_label:
                if a.sentiment_label.value == 'positive':
                    pos += 1
                elif a.sentiment_label.value == 'negative':
                    neg += 1
                else:
                    neu += 1
        for s in social_rows:
            if s.sentiment_label:
                if s.sentiment_label.value == 'positive':
                    pos += 1
                elif s.sentiment_label.value == 'negative':
                    neg += 1
                else:
                    neu += 1

        total = pos + neg + neu if (pos + neg + neu) > 0 else 1

        return {
            'sku': product.sku,
            'product_name': product.name,
            'keywords': [k for k,_ in top],
            'counts': {k: int(v) for k,v in top},
            'sentiment_summary': {
                'positive': pos,
                'negative': neg,
                'neutral': neu,
                'positive_pct': round(pos/total*100,1),
                'negative_pct': round(neg/total*100,1),
                'neutral_pct': round(neu/total*100,1)
            }
        }
    
    def save_bulk_news_articles(self, db: Session, articles_data: List[Dict]) -> int:
        """Save multiple news articles efficiently"""
        saved_count = 0
        
        for article_data in articles_data:
            try:
                # Check if article already exists
                existing = db.query(NewsArticle).filter(
                    NewsArticle.article_id == article_data.get('article_id')
                ).first()
                
                if not existing:
                    save_news_article(db, article_data)
                    saved_count += 1
                    
            except Exception as e:
                print(f"Error saving article: {e}")
                continue
        
        return saved_count
    
    def update_stock_sentiment(self, db: Session, symbol: str, sentiment_data: Dict):
        """Update sentiment data for a product (SKU) - legacy method name kept"""
        # Legacy wrapper for compatibility
        return self.update_product_sentiment(db, symbol, sentiment_data)

    def update_product_sentiment(self, db: Session, sku: str, sentiment_data: Dict):
        """Update sentiment data for a product (SKU)"""
        product = get_or_create_product(db, sku)
        save_sentiment_snapshot(db, product.id, sentiment_data)
        return product
    
    def cleanup_old_data(self, db: Session, days_to_keep: int = 30):
        """Clean up old data to maintain database performance"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Delete old sentiment snapshots
        old_snapshots = db.query(SentimentSnapshot).filter(
            SentimentSnapshot.snapshot_time < cutoff_date
        ).delete()
        
        # Delete old news articles
        old_articles = db.query(NewsArticle).filter(
            NewsArticle.published_at < cutoff_date
        ).delete()
        
        # Delete old economic events
        old_events = db.query(EconomicEvent).filter(
            EconomicEvent.event_date < cutoff_date
        ).delete()
        
        # Delete old refresh logs
        old_logs = db.query(DataRefreshLog).filter(
            DataRefreshLog.started_at < cutoff_date
        ).delete()
        
        db.commit()
        
        return {
            'snapshots_deleted': old_snapshots,
            'articles_deleted': old_articles,
            'events_deleted': old_events,
            'logs_deleted': old_logs
        }
    
    def get_data_refresh_status(self, db: Session) -> Dict:
        """Get status of recent data refresh operations"""
        recent_logs = db.query(DataRefreshLog).filter(
            DataRefreshLog.started_at >= datetime.utcnow() - timedelta(hours=24)
        ).order_by(desc(DataRefreshLog.started_at)).limit(10).all()
        
        return {
            'recent_refreshes': [
                {
                    'data_source': log.data_source,
                    'status': log.status,
                    'records_processed': log.records_processed,
                    'started_at': log.started_at,
                    'completed_at': log.completed_at,
                    'duration_seconds': log.duration_seconds,
                    'error_message': log.error_message
                }
                for log in recent_logs
            ]
        }
    
    def _time_ago(self, timestamp: datetime) -> str:
        """Convert timestamp to human-readable time ago format"""
        if not timestamp:
            return 'Unknown'
        
        now = datetime.utcnow()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"

# Global service instance
db_service = DatabaseService()
