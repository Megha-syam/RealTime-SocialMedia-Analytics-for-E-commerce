"""
Market-specific sentiment analysis and trend detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
from dataclasses import dataclass

@dataclass
class MarketEvent:
    """Represents a significant market event"""
    event_type: str
    description: str
    timestamp: datetime
    sentiment_impact: float
    confidence: float
    related_symbols: List[str]

class MarketSentimentAnalyzer:
    """
    Specialized analyzer for market-wide sentiment trends and events
    """
    
    def __init__(self):
        # Market event patterns
        self.event_patterns = {
            'earnings': [
                r'earnings\s+(beat|miss|exceed|disappoint)',
                r'quarterly\s+results',
                r'eps\s+(beat|miss)',
                r'revenue\s+(growth|decline|beat|miss)'
            ],
            'fed_policy': [
                r'federal\s+reserve',
                r'interest\s+rate',
                r'fed\s+(meeting|decision|policy)',
                r'monetary\s+policy',
                r'jerome\s+powell'
            ],
            'economic_data': [
                r'gdp\s+(growth|decline)',
                r'unemployment\s+rate',
                r'inflation\s+(data|rate)',
                r'consumer\s+price\s+index',
                r'non.?farm\s+payrolls'
            ],
            'market_structure': [
                r'market\s+(crash|correction|rally)',
                r'bull\s+market',
                r'bear\s+market',
                r'volatility\s+(spike|surge)',
                r'vix\s+(above|below)'
            ],
            'geopolitical': [
                r'trade\s+(war|deal|agreement)',
                r'geopolitical\s+(tension|risk)',
                r'sanctions',
                r'brexit',
                r'china\s+(trade|tariff)'
            ]
        }
        
        # Sector-specific keywords
        self.sector_keywords = {
            'technology': ['ai', 'artificial intelligence', 'cloud', 'software', 'semiconductor', 'chip'],
            'healthcare': ['drug', 'pharmaceutical', 'biotech', 'clinical trial', 'fda approval'],
            'energy': ['oil', 'gas', 'renewable', 'solar', 'wind', 'crude'],
            'financial': ['bank', 'lending', 'credit', 'mortgage', 'fintech'],
            'consumer': ['retail', 'consumer spending', 'e-commerce', 'brand'],
            'industrial': ['manufacturing', 'supply chain', 'logistics', 'infrastructure']
        }
        
        # Sentiment momentum indicators
        self.momentum_window = 24  # hours
        self.trend_threshold = 0.15  # minimum change for trend detection
        
    def analyze_market_events(self, articles: List[Dict], 
                            time_window_hours: int = 24) -> List[MarketEvent]:
        """
        Detect and analyze significant market events from news articles
        """
        events = []
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Group articles by event type
        event_groups = defaultdict(list)
        
        for article in articles:
            # Skip old articles
            pub_time = article.get('published_at')
            if pub_time and isinstance(pub_time, str):
                pub_time = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
            if pub_time and pub_time < cutoff_time:
                continue
            
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            
            # Classify article by event type
            for event_type, patterns in self.event_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        event_groups[event_type].append(article)
                        break
        
        # Analyze each event group
        for event_type, group_articles in event_groups.items():
            if len(group_articles) >= 2:  # Minimum threshold for significance
                event = self._analyze_event_group(event_type, group_articles)
                if event:
                    events.append(event)
        
        # Sort by impact and recency
        events.sort(key=lambda x: (abs(x.sentiment_impact), x.timestamp), reverse=True)
        
        return events[:10]  # Return top 10 events
    
    def detect_sentiment_trends(self, sentiment_history: List[Dict]) -> Dict:
        """
        Detect sentiment trends and momentum from historical data
        """
        if len(sentiment_history) < 5:
            return {'trend': 'insufficient_data', 'momentum': 0.0, 'confidence': 0.0}
        
        # Convert to pandas for easier analysis
        df = pd.DataFrame(sentiment_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate moving averages
        df['ma_short'] = df['sentiment'].rolling(window=3).mean()
        df['ma_long'] = df['sentiment'].rolling(window=6).mean()
        
        # Calculate momentum
        recent_sentiment = df['sentiment'].tail(3).mean()
        older_sentiment = df['sentiment'].head(3).mean()
        momentum = recent_sentiment - older_sentiment
        
        # Determine trend
        latest_short_ma = df['ma_short'].iloc[-1]
        latest_long_ma = df['ma_long'].iloc[-1]
        
        if latest_short_ma > latest_long_ma + self.trend_threshold:
            trend = 'bullish'
        elif latest_short_ma < latest_long_ma - self.trend_threshold:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Calculate trend confidence
        sentiment_std = df['sentiment'].std()
        confidence = max(0.0, 1.0 - sentiment_std)
        
        # Detect volatility
        volatility = df['sentiment'].rolling(window=5).std().iloc[-1]
        
        return {
            'trend': trend,
            'momentum': round(momentum, 4),
            'confidence': round(confidence, 4),
            'volatility': round(volatility, 4),
            'current_sentiment': round(recent_sentiment, 4),
            'sentiment_change_24h': round(momentum, 4),
            'trend_strength': min(1.0, abs(momentum) / self.trend_threshold)
        }
    
    def analyze_sector_sentiment(self, articles: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze sentiment by market sector
        """
        sector_articles = defaultdict(list)
        
        # Classify articles by sector
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            
            for sector, keywords in self.sector_keywords.items():
                if any(keyword in text for keyword in keywords):
                    sector_articles[sector].append(article)
        
        # Analyze sentiment for each sector
        sector_analysis = {}
        
        for sector, articles_list in sector_articles.items():
            if len(articles_list) >= 2:
                sentiments = []
                for article in articles_list:
                    # Use existing sentiment score or calculate
                    sentiment = article.get('sentiment_score', 0.0)
                    sentiments.append(sentiment)
                
                avg_sentiment = np.mean(sentiments)
                sentiment_std = np.std(sentiments)
                
                sector_analysis[sector] = {
                    'average_sentiment': round(avg_sentiment, 4),
                    'volatility': round(sentiment_std, 4),
                    'article_count': len(articles_list),
                    'sentiment_label': self._get_sentiment_label(avg_sentiment),
                    'confidence': max(0.0, 1.0 - sentiment_std)
                }
        
        return sector_analysis
    
    def calculate_fear_greed_index(self, market_data: Dict) -> Dict:
        """
        Calculate a Fear & Greed index based on multiple market indicators
        """
        indicators = {}
        
        # Market momentum (25% weight)
        momentum = market_data.get('momentum', 0.0)
        momentum_score = max(0, min(100, (momentum + 1) * 50))
        indicators['momentum'] = momentum_score
        
        # Market volatility (25% weight) - inverted (low volatility = greed)
        volatility = market_data.get('volatility', 0.5)
        volatility_score = max(0, min(100, (1 - volatility) * 100))
        indicators['volatility'] = volatility_score
        
        # News sentiment (25% weight)
        news_sentiment = market_data.get('news_sentiment', 0.0)
        news_score = max(0, min(100, (news_sentiment + 1) * 50))
        indicators['news_sentiment'] = news_score
        
        # Economic indicators (25% weight)
        economic_sentiment = market_data.get('economic_sentiment', 0.0)
        economic_score = max(0, min(100, (economic_sentiment + 1) * 50))
        indicators['economic_sentiment'] = economic_score
        
        # Calculate weighted average
        fear_greed_score = (
            indicators['momentum'] * 0.25 +
            indicators['volatility'] * 0.25 +
            indicators['news_sentiment'] * 0.25 +
            indicators['economic_sentiment'] * 0.25
        )
        
        # Determine label
        if fear_greed_score >= 75:
            label = "Extreme Greed"
        elif fear_greed_score >= 55:
            label = "Greed"
        elif fear_greed_score >= 45:
            label = "Neutral"
        elif fear_greed_score >= 25:
            label = "Fear"
        else:
            label = "Extreme Fear"
        
        return {
            'score': round(fear_greed_score, 1),
            'label': label,
            'indicators': indicators,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _analyze_event_group(self, event_type: str, articles: List[Dict]) -> Optional[MarketEvent]:
        """Analyze a group of articles for a specific event type"""
        if not articles:
            return None
        
        # Calculate aggregate sentiment
        sentiments = []
        confidences = []
        symbols = set()
        
        for article in articles:
            sentiment = article.get('sentiment_score', 0.0)
            confidence = article.get('confidence', 0.5)
            
            sentiments.append(sentiment)
            confidences.append(confidence)
            
            # Extract potential stock symbols (basic regex)
            text = article.get('title', '') + ' ' + article.get('description', '')
            found_symbols = re.findall(r'\b[A-Z]{2,5}\b', text)
            symbols.update(found_symbols)
        
        avg_sentiment = np.mean(sentiments)
        avg_confidence = np.mean(confidences)
        
        # Get most recent timestamp
        timestamps = []
        for article in articles:
            pub_time = article.get('published_at')
            if pub_time:
                if isinstance(pub_time, str):
                    pub_time = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                timestamps.append(pub_time)
        
        latest_time = max(timestamps) if timestamps else datetime.utcnow()
        
        # Create event description
        description = f"{event_type.replace('_', ' ').title()} event detected from {len(articles)} articles"
        
        return MarketEvent(
            event_type=event_type,
            description=description,
            timestamp=latest_time,
            sentiment_impact=avg_sentiment,
            confidence=avg_confidence,
            related_symbols=list(symbols)[:5]  # Limit to 5 symbols
        )
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert numerical score to categorical label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'

# Global market analyzer instance
market_sentiment_analyzer = MarketSentimentAnalyzer()
