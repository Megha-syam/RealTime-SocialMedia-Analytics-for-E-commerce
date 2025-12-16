from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import httpx
from datetime import datetime, timedelta
import os
from contextlib import asynccontextmanager
from backend.database_service import db_service
from backend.realtime_data_manager import data_manager

try:
    from sentiment_analysis import financial_sentiment_analyzer, market_sentiment_analyzer
    from api_clients import news_client, economics_client, ai_client
except ImportError:
    # Fallback for when modules don't exist
    financial_sentiment_analyzer = None
    market_sentiment_analyzer = None
    news_client = None
    economics_client = None
    ai_client = None

# API Configuration
NEWS_API_KEY = "pub_f6cb8108c3dd449299f965cd7c131702"
ECONOMIC_API_KEY = "b4a2da19-2dec-4a25-a97e-290d452a91d4"
GEMINI_API_KEY = "AIzaSyCnvqMspISmF_IAFkTO89czQkRxoKHLQzY"

sentiment_cache = {}
news_cache = []
economic_cache = []
recommendations_cache = {}
last_update = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start periodic fetchers and the real-time data manager (Reddit ingestion + in-memory trending)
    asyncio.create_task(periodic_data_fetch())
    # start the RealTimeDataManager background monitoring (will start Reddit thread if enabled)
    try:
        asyncio.create_task(data_manager.start_realtime_monitoring())
    except Exception as e:
        print(f"Warning: failed to start realtime data manager: {e}")
    yield
    # Shutdown: cleanup if needed

app = FastAPI(
    title="Real-Time Social Media Analytics - E-Commerce Trends Backend",
    description="Real-time social media sentiment and trend analytics API focused on e-commerce marketplaces",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SentimentRequest(BaseModel):
    symbol: str
    timeframe: Optional[str] = "1d"

class SentimentResponse(BaseModel):
    symbol: str
    overall_sentiment: float
    news_sentiment: float
    social_sentiment: float
    economic_sentiment: float
    confidence: float
    last_updated: datetime

class NewsItem(BaseModel):
    id: str
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    sentiment_label: str

class RecommendationResponse(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    risk_level: str

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Real-Time Social Media Analytics for E-Commerce Trends API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "last_data_update": last_update
    }

@app.get("/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """Get real-time sentiment analysis for a product SKU.

    Note: parameter name `symbol` is kept for backward compatibility with the
    frontend; internally this refers to a product SKU.
    """
    try:
        sentiment_data = await fetch_realtime_sentiment(symbol)

        if not sentiment_data:
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {symbol}")

        trend_analysis = calculate_trend_from_cache(symbol)

        return {
            "symbol": symbol,
            "current_sentiment": sentiment_data,
            "trend_analysis": trend_analysis,
            "last_updated": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sentiment: {str(e)}")

@app.get("/news")
async def get_news(limit: int = 10, category: str = "business"):
    """Get latest e-commerce/retail news with real-time sentiment analysis"""
    try:
        # Refresh upstream news cache
        await fetch_news_data(category)

        # Combine primary news cache with in-memory realtime news from data_manager (e.g., Reddit)
        combined = list(news_cache)
        try:
            with data_manager._data_lock:
                for a in data_manager.news_data:
                    # normalize to dict shape similar to news_cache items
                    item = {
                        "id": a.id,
                        "title": a.title,
                        "description": a.description,
                        "source": a.source,
                        "url": a.url,
                        "published_at": a.published_at.isoformat() if hasattr(a.published_at, 'isoformat') else str(a.published_at),
                        "sentiment_score": a.sentiment_score,
                        "sentiment_label": a.sentiment_label,
                    }
                    combined.append(item)
        except Exception:
            # If data_manager not ready, ignore and return news_cache only
            pass

        # Deduplicate by url/id keeping newest
        seen = set()
        deduped = []
        for item in sorted(combined, key=lambda x: x.get('published_at', ''), reverse=True):
            key = item.get('url') or item.get('id')
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return {"news": deduped[:limit], "total": len(deduped)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

@app.get("/recommendations/{symbol}")
async def get_recommendations(symbol: str):
    """Get real-time AI-powered recommendations for a product (promotions/pricing/restock).

    The API keeps the path param named `symbol` for compatibility; pass a SKU.
    The recommendation action values (BUY/SELL/HOLD) are retained for UI compatibility
    but represent e-commerce actions (e.g., BUY -> promote/raise price, SELL -> discount/clearance, HOLD -> no action).
    """
    try:
        recommendation = await generate_realtime_recommendation(symbol)
        return recommendation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.get("/economic-events")
async def get_economic_events():
    """Get real-time economic events and their impact"""
    try:
        await fetch_economic_data()
        
        return {"events": economic_cache, "total": len(economic_cache)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching economic data: {str(e)}")

@app.get("/market-analysis")
async def get_market_analysis():
    """Get real-time comprehensive market sentiment analysis"""
    try:
        await fetch_all_data()  # Ensure fresh data
        
        # Analyze market events from current news
        market_events = market_sentiment_analyzer.analyze_market_events(news_cache)
        
        # Get sector analysis from current news
        sector_analysis = market_sentiment_analyzer.analyze_sector_sentiment(news_cache)
        
        # Calculate market overview from current data
        market_overview = calculate_market_overview()
        
        # Calculate Fear & Greed Index from current sentiment
        fear_greed = market_sentiment_analyzer.calculate_fear_greed_index({
            'momentum': market_overview.get('overall_sentiment', 50) / 50 - 1,
            'volatility': 0.3,
            'news_sentiment': sum(a.get('sentiment_score', 0) for a in news_cache) / len(news_cache) if news_cache else 0,
            'economic_sentiment': sum(e.get('sentiment_score', 0) for e in economic_cache) / len(economic_cache) if economic_cache else 0
        })
        
        return {
            "market_overview": market_overview,
            "market_events": [
                {
                    "event_type": event.event_type,
                    "description": event.description,
                    "timestamp": event.timestamp,
                    "sentiment_impact": event.sentiment_impact,
                    "confidence": event.confidence,
                    "related_symbols": event.related_symbols
                }
                for event in market_events
            ],
            "sector_analysis": sector_analysis,
            "fear_greed_index": fear_greed,
            "last_updated": datetime.utcnow()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market analysis: {str(e)}")


@app.get("/trending/{symbol}")
async def get_trending_keywords(symbol: str, top_n: int = 10):
    """Return trending e-commerce keywords for a SKU/product (keeps `symbol` for compatibility)"""
    try:
        # Use database service to compute trending keywords
        from backend.database import SessionLocal
        db = SessionLocal()
        try:
            result = db_service.get_trending_keywords_for_sku(db, symbol, top_n=top_n)
            return JSONResponse(content={"trending": result})
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing trending keywords: {e}")


@app.get("/trending-live/{symbol}")
async def get_trending_live(symbol: str, top_n: int = 10):
    """Return live trending keywords from in-memory news (no DB required)."""
    try:
        # Use the global data_manager to compute live trending keywords
        result = data_manager.get_live_trending_keywords(symbol, top_n=top_n)
        return JSONResponse(content={"trending_live": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing live trending keywords: {e}")

@app.post("/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """Manually trigger data refresh"""
    background_tasks.add_task(fetch_all_data)
    return {"message": "Data refresh initiated", "timestamp": datetime.now()}

# Data fetching functions
async def fetch_news_data(category: str = "business"):
    """Fetch news from News API"""
    global news_cache, last_update
    
    try:
        async with httpx.AsyncClient() as client:
            # News API endpoint
            url = "https://newsdata.io/api/1/news"
            params = {
                "apikey": NEWS_API_KEY,
                "category": category,
                "language": "en",
                "country": "us",
                "size": 50
            }
            
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Process and analyze sentiment for each article
            processed_news = []
            for article in data.get("results", []):
                if article.get("title") and article.get("description"):
                    sentiment_score, sentiment_label = analyze_text_sentiment(
                        f"{article['title']} {article['description']}"
                    )
                    
                    processed_news.append({
                        "id": article.get("article_id", ""),
                        "title": article["title"],
                        "description": article.get("description", ""),
                        "source": article.get("source_id", "Unknown"),
                        "url": article.get("link", ""),
                        "published_at": article.get("pubDate", datetime.now().isoformat()),
                        "sentiment_score": sentiment_score,
                        "sentiment_label": sentiment_label
                    })
            
            news_cache = processed_news
            last_update = datetime.now()
            
    except Exception as e:
        print(f"Error fetching news data: {e}")

async def fetch_economic_data():
    """Fetch economic events from Trading Economics API"""
    global economic_cache, last_update
    
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://api.tradingeconomics.com/calendar?c={ECONOMIC_API_KEY}"
            
            response = await client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Process economic events
            processed_events = []
            for event in data[:20]:  # Limit to recent events
                if event.get("Event") and event.get("Country"):
                    # Analyze impact sentiment
                    impact_text = f"{event.get('Event', '')} {event.get('Country', '')}"
                    sentiment_score, sentiment_label = analyze_text_sentiment(impact_text)
                    
                    processed_events.append({
                        "event": event.get("Event", ""),
                        "country": event.get("Country", ""),
                        "date": event.get("Date", ""),
                        "actual": event.get("Actual", ""),
                        "forecast": event.get("Forecast", ""),
                        "previous": event.get("Previous", ""),
                        "importance": event.get("Importance", "Low"),
                        "sentiment_score": sentiment_score,
                        "sentiment_label": sentiment_label
                    })
            
            economic_cache = processed_events
            last_update = datetime.now()
            
    except Exception as e:
        print(f"Error fetching economic data: {e}")

async def fetch_realtime_sentiment(symbol: str) -> dict:
    """Fetch and calculate real-time sentiment for a SKU (public param name `symbol` kept for compatibility)"""
    try:
        # Map public parameter name to internal SKU
        sku = symbol
        # Get fresh news for the SKU/product
        symbol_news = await fetch_symbol_specific_news(sku)
        
        # Analyze sentiment for each article
        news_sentiments = []
        for article in symbol_news:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            # Use product-focused sentiment analyzer if available
            if financial_sentiment_analyzer:
                sentiment_result = financial_sentiment_analyzer.analyze_text(text, context='news')
                news_sentiments.append(sentiment_result.get('composite_score', 0))
            else:
                s, _ = analyze_text_sentiment(text)
                news_sentiments.append(s)
        
        # Calculate aggregated sentiment
        news_sentiment = sum(news_sentiments) / len(news_sentiments) if news_sentiments else 0.0
        social_sentiment = 0.23  # Placeholder for social media sentiment
        economic_sentiment = calculate_economic_sentiment()
        
        # Weighted average
        overall_sentiment = (
            news_sentiment * 0.4 +
            social_sentiment * 0.3 +
            economic_sentiment * 0.3
        )
        
        # Normalize to 0-100 scale
        overall_sentiment_score = max(0, min(100, (overall_sentiment + 1) * 50))
        
        sentiment_data = {
            "overall_sentiment": overall_sentiment_score,
            "news_sentiment": (news_sentiment + 1) * 50,
            "social_sentiment": (social_sentiment + 1) * 50,
            "economic_sentiment": (economic_sentiment + 1) * 50,
            "confidence": 0.85,
            "news_volume": len(news_sentiments),
            "social_volume": 0,
            "economic_events_count": len(economic_cache),
            "timestamp": datetime.utcnow()
        }
        
        if sku not in sentiment_cache:
            sentiment_cache[sku] = []
        sentiment_cache[sku].append(sentiment_data)
        
        # Keep only last 24 hours of data
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        sentiment_cache[sku] = [
            s for s in sentiment_cache[sku] 
            if s['timestamp'] > cutoff_time
        ]

        return sentiment_data

    except Exception as e:
        print(f"Error calculating real-time sentiment for {symbol}: {e}")
        return None

async def fetch_symbol_specific_news(symbol: str) -> list:
    """Fetch news specifically mentioning the SKU/product (param name `symbol` kept for compatibility)"""
    try:
        async with httpx.AsyncClient() as client:
            url = "https://newsdata.io/api/1/news"
            params = {
                "apikey": NEWS_API_KEY,
                # Search for SKU or product name mentions
                "q": symbol,
                "language": "en",
                "category": "business",
                "size": 20
            }
            
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
            
    except Exception as e:
        print(f"Error fetching symbol-specific news: {e}")
        return []

async def generate_realtime_recommendation(symbol: str) -> dict:
    """Generate real-time AI recommendation for a SKU (keeps `symbol` path param for compatibility)."""
    try:
        # Use in-memory realtime news (including Reddit) from data_manager to build context
        sku = symbol
        with data_manager._data_lock:
            articles = [a for a in data_manager.news_data if (sku in (a.skus_mentioned or [])) or (sku.lower() in (a.title or '').lower()) or (sku.lower() in (a.description or '').lower())]

        # If no recent articles found, fallback to neutral HOLD recommendation
        if not articles:
            recommendation = {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 50,
                "reasoning": "No recent social or news mentions found for this product.",
                "risk_level": "Low",
                "timestamp": datetime.utcnow(),
                "data_source": "REALTIME_REDDIT",
            }
            recommendations_cache[symbol] = recommendation
            return recommendation

        # Compose combined text and compute simple sentiment using data_manager's analyzer
        combined_text = " \n ".join([f"{a.title} {a.description}" for a in articles])
        try:
            score, label = data_manager.analyze_text_sentiment(combined_text)
        except Exception:
            # if analyzer unavailable, approximate with averaging article scores
            scores = [a.sentiment_score for a in articles if hasattr(a, 'sentiment_score')]
            score = sum(scores) / len(scores) if scores else 0.0
            label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")

        # Map sentiment to e-commerce action
        if score > 0.1:
            action_label = "PROMOTE"
            mapped_action = "BUY"
        elif score < -0.1:
            action_label = "DISCOUNT"
            mapped_action = "SELL"
        else:
            action_label = "HOLD"
            mapped_action = "HOLD"

        # Confidence scaled from sentiment magnitude and article count
        confidence = int(min(95, max(40, abs(score) * 100 + min(30, len(articles) * 5))))

        reasoning = f"Based on {len(articles)} recent mentions with predominant sentiment '{label}'."

        recommendation = {
            "symbol": symbol,
            "action": mapped_action,
            "raw_action": action_label,
            "confidence": confidence,
            "reasoning": reasoning,
            "risk_level": "Medium",
            "timestamp": datetime.utcnow(),
            "data_source": "REALTIME_REDDIT",
        }

        recommendations_cache[symbol] = recommendation
        return recommendation
    except Exception as e:
        print(f"Error generating real-time recommendation: {e}")
        return {
            "symbol": symbol,
            "action": "HOLD",
            "confidence": 50,
            "reasoning": "Unable to generate recommendation due to internal error.",
            "risk_level": "High",
            "timestamp": datetime.utcnow(),
        }

def calculate_trend_from_cache(symbol: str) -> dict:
    """Calculate sentiment trend from cached data"""
    if symbol not in sentiment_cache or len(sentiment_cache[symbol]) < 2:
        return {"trend": "neutral", "change": 0.0}
    
    recent_data = sentiment_cache[symbol][-10:]  # Last 10 data points
    
    if len(recent_data) >= 2:
        current_sentiment = recent_data[-1]['overall_sentiment']
        previous_sentiment = recent_data[-2]['overall_sentiment']
        change = current_sentiment - previous_sentiment
        
        if change > 5:
            trend = "bullish"
        elif change < -5:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {"trend": trend, "change": change}
    
    return {"trend": "neutral", "change": 0.0}

def calculate_market_overview() -> dict:
    """Calculate market overview from current cached data"""
    if not news_cache:
        return {"overall_sentiment": 50, "market_mood": "neutral"}
    
    total_sentiment = sum(article.get('sentiment_score', 0) for article in news_cache)
    avg_sentiment = total_sentiment / len(news_cache)
    
    # Convert to 0-100 scale
    overall_sentiment = max(0, min(100, (avg_sentiment + 1) * 50))
    
    if overall_sentiment > 60:
        mood = "bullish"
    elif overall_sentiment < 40:
        mood = "bearish"
    else:
        mood = "neutral"
    
    return {
        "overall_sentiment": overall_sentiment,
        "market_mood": mood,
        "news_volume": len(news_cache),
        "economic_events": len(economic_cache)
    }

async def fetch_all_data():
    """Fetch all data sources"""
    await asyncio.gather(
        fetch_news_data(),
        fetch_economic_data(),
    )

async def periodic_data_fetch():
    """Background task to fetch data every 30 minutes"""
    while True:
        try:
            await fetch_all_data()
            print(f"Real-time data refreshed at {datetime.now()}")
        except Exception as e:
            print(f"Error in periodic data fetch: {e}")
        
        await asyncio.sleep(900)  # 15 minutes

def analyze_text_sentiment(text: str) -> tuple[float, str]:
    """Basic sentiment analysis using VADER (placeholder implementation)"""
    # In production, use proper sentiment analysis libraries
    # This is a simplified version for demonstration
    
    positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'growth', 'profit', 'gain', 'rise', 'up']
    negative_words = ['bad', 'terrible', 'negative', 'bearish', 'loss', 'decline', 'fall', 'down', 'crash', 'drop']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        score = min(0.8, positive_count * 0.2)
        label = "positive"
    elif negative_count > positive_count:
        score = max(-0.8, -negative_count * 0.2)
        label = "negative"
    else:
        score = 0.0
        label = "neutral"
    
    return score, label

def calculate_news_sentiment(symbol: str) -> float:
    """Calculate average sentiment from news mentioning the symbol"""
    relevant_news = [
        article for article in news_cache 
        if symbol.lower() in article.get("title", "").lower() or 
           symbol.lower() in article.get("description", "").lower()
    ]
    
    if not relevant_news:
        return 0.0
    
    total_sentiment = sum(article.get("sentiment_score", 0) for article in relevant_news)
    return total_sentiment / len(relevant_news)

def calculate_economic_sentiment() -> float:
    """Calculate sentiment from economic events"""
    if not economic_cache:
        return 0.0
    
    total_sentiment = sum(event.get("sentiment_score", 0) for event in economic_cache)
    return total_sentiment / len(economic_cache)

def get_recent_headlines_for_symbol(symbol: str) -> str:
    """Get recent headlines mentioning the symbol"""
    relevant_news = [
        article for article in news_cache[:5]
        if symbol.lower() in article.get("title", "").lower()
    ]
    
    headlines = [article.get("title", "") for article in relevant_news]
    return "; ".join(headlines[:3])

def is_cache_stale() -> bool:
    """Check if cache needs refresh (30 minutes)"""
    if not last_update:
        return True
    return datetime.now() - last_update > timedelta(minutes=30)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
