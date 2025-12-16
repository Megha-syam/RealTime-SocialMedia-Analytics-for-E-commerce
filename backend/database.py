"""
Database connection and ORM models using SQLAlchemy
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, Enum, ForeignKey, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum
import os
from typing import Optional

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/ecommerce_social_analytics")

# Create engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enum definitions
class SentimentLabel(enum.Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class RecommendationAction(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class RiskLevel(enum.Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"

# Database Models
class Product(Base):
    __tablename__ = "products"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sku = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255))
    category = Column(String(100))
    market_cap = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sentiment_snapshots = relationship("SentimentSnapshot", back_populates="product", cascade="all, delete-orphan")
    ai_recommendations = relationship("AIRecommendation", back_populates="product", cascade="all, delete-orphan")
    social_mentions = relationship("SocialMention", back_populates="product", cascade="all, delete-orphan")

# Backwards compatibility alias: keep `Stock` name referring to Product
Stock = Product

class NewsArticle(Base):
    __tablename__ = "news_articles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(String(255), unique=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    content = Column(Text)
    source = Column(String(100))
    author = Column(String(255))
    url = Column(Text)
    published_at = Column(DateTime)
    sentiment_score = Column(Float)  # -1.0 to 1.0
    sentiment_label = Column(Enum(SentimentLabel))
    confidence = Column(Float)  # 0.0 to 1.0
    vader_score = Column(Float)
    textblob_score = Column(Float)
    financial_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EconomicEvent(Base):
    __tablename__ = "economic_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_name = Column(String(255), nullable=False)
    country = Column(String(100))
    event_date = Column(DateTime)
    actual_value = Column(String(50))
    forecast_value = Column(String(50))
    previous_value = Column(String(50))
    importance = Column(String(20))
    currency = Column(String(3))
    sentiment_score = Column(Float)
    sentiment_label = Column(Enum(SentimentLabel))
    impact_level = Column(Integer)  # 1-5 scale
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SocialMention(Base):
    __tablename__ = "social_mentions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    platform = Column(String(50))
    mention_id = Column(String(255))
    content = Column(Text)
    author = Column(String(255))
    followers_count = Column(Integer)
    likes_count = Column(Integer)
    retweets_count = Column(Integer)
    posted_at = Column(DateTime)
    sentiment_score = Column(Float)
    sentiment_label = Column(Enum(SentimentLabel))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="social_mentions")

class SentimentSnapshot(Base):
    __tablename__ = "sentiment_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    snapshot_time = Column(DateTime, nullable=False)
    overall_sentiment = Column(Float)  # 0.0 to 100.0
    news_sentiment = Column(Float)
    social_sentiment = Column(Float)
    economic_sentiment = Column(Float)
    confidence = Column(Float)
    news_volume = Column(Integer, default=0)
    social_volume = Column(Integer, default=0)
    economic_events_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="sentiment_snapshots")

class AIRecommendation(Base):
    __tablename__ = "ai_recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    recommendation_action = Column(Enum(RecommendationAction), nullable=False)
    confidence = Column(Integer)  # 0-100
    reasoning = Column(Text)
    risk_level = Column(Enum(RiskLevel))
    time_horizon = Column(String(20))
    generated_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="ai_recommendations")

class MarketIndicator(Base):
    __tablename__ = "market_indicators"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    indicator_name = Column(String(100), nullable=False)
    country = Column(String(100))
    value = Column(Float)
    unit = Column(String(20))
    date = Column(DateTime)
    previous_value = Column(Float)
    change_percent = Column(Float)
    sentiment_impact = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DataRefreshLog(Base):
    __tablename__ = "data_refresh_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_source = Column(String(50), nullable=False)
    refresh_type = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    records_processed = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)

# Database utility functions
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_product_by_sku(db, sku: str) -> Optional[Product]:
    """Get product by SKU"""
    return db.query(Product).filter(Product.sku == sku.upper()).first()


# Backwards-compatible wrapper
def get_stock_by_symbol(db, symbol: str) -> Optional[Stock]:
    """Legacy wrapper: Get stock by symbol (alias for get_product_by_sku)"""
    return get_product_by_sku(db, symbol)

def get_or_create_product(db, sku: str, name: str = None, category: str = None) -> Product:
    """Get existing product or create new one"""
    product = get_product_by_sku(db, sku)
    if not product:
        product = Product(
            sku=sku.upper(),
            name=name,
            category=category
        )
        db.add(product)
        db.commit()
        db.refresh(product)
    return product


# Backwards-compatible wrapper
def get_or_create_stock(db, symbol: str, name: str = None, sector: str = None) -> Stock:
    """Legacy wrapper: Get or create stock (alias for get_or_create_product)"""
    return get_or_create_product(db, symbol, name=name, category=sector)

def save_sentiment_snapshot(db, product_id: uuid.UUID, sentiment_data: dict):
    """Save sentiment snapshot to database (product-aware)"""
    snapshot = SentimentSnapshot(
        product_id=product_id,
        snapshot_time=datetime.utcnow(),
        overall_sentiment=sentiment_data.get('overall_sentiment', 0),
        news_sentiment=sentiment_data.get('news_sentiment', 0),
        social_sentiment=sentiment_data.get('social_sentiment', 0),
        economic_sentiment=sentiment_data.get('economic_sentiment', 0),
        confidence=sentiment_data.get('confidence', 0),
        news_volume=sentiment_data.get('news_volume', 0),
        social_volume=sentiment_data.get('social_volume', 0),
        economic_events_count=sentiment_data.get('economic_events_count', 0)
    )
    db.add(snapshot)
    db.commit()
    return snapshot

def save_news_article(db, article_data: dict):
    """Save news article with sentiment analysis"""
    # Check if article already exists
    existing = db.query(NewsArticle).filter(
        NewsArticle.article_id == article_data.get('article_id')
    ).first()
    
    if existing:
        return existing
    
    article = NewsArticle(
        article_id=article_data.get('article_id'),
        title=article_data.get('title'),
        description=article_data.get('description'),
        content=article_data.get('content'),
        source=article_data.get('source'),
        author=article_data.get('author'),
        url=article_data.get('url'),
        published_at=article_data.get('published_at'),
        sentiment_score=article_data.get('sentiment_score'),
        sentiment_label=SentimentLabel(article_data.get('sentiment_label', 'neutral')),
        confidence=article_data.get('confidence'),
        vader_score=article_data.get('vader_score'),
        textblob_score=article_data.get('textblob_score'),
        financial_score=article_data.get('financial_score')
    )
    db.add(article)
    db.commit()
    return article

def save_ai_recommendation(db, product_id: uuid.UUID, recommendation_data: dict):
    """Save AI recommendation for a product"""
    # Deactivate previous recommendations
    db.query(AIRecommendation).filter(
        AIRecommendation.product_id == product_id,
        AIRecommendation.is_active == True
    ).update({'is_active': False})

    recommendation = AIRecommendation(
        product_id=product_id,
        recommendation_action=RecommendationAction(recommendation_data.get('action', 'HOLD')),
        confidence=recommendation_data.get('confidence', 50),
        reasoning=recommendation_data.get('reasoning'),
        risk_level=RiskLevel(recommendation_data.get('risk_level', 'Medium')),
        time_horizon=recommendation_data.get('time_horizon', 'Medium'),
        generated_at=datetime.utcnow(),
        expires_at=datetime.utcnow().replace(hour=23, minute=59, second=59),  # Expire at end of day
        is_active=True
    )
    db.add(recommendation)
    db.commit()
    return recommendation

def log_data_refresh(db, source: str, refresh_type: str, status: str, 
                    records_processed: int = 0, error_message: str = None,
                    started_at: datetime = None, completed_at: datetime = None):
    """Log data refresh operation"""
    if not started_at:
        started_at = datetime.utcnow()
    
    duration = None
    if completed_at and started_at:
        duration = int((completed_at - started_at).total_seconds())
    
    log_entry = DataRefreshLog(
        data_source=source,
        refresh_type=refresh_type,
        status=status,
        records_processed=records_processed,
        error_message=error_message,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration
    )
    db.add(log_entry)
    db.commit()
    return log_entry
