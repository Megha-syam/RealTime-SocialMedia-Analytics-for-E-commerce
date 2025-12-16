"""
Advanced sentiment analysis module using multiple approaches
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from typing import Dict, Tuple
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Financial-specific keywords
        self.financial_positive = [
            'bullish', 'rally', 'surge', 'gains', 'profit', 'growth', 'earnings beat',
            'upgrade', 'outperform', 'buy rating', 'strong performance', 'revenue growth',
            'market leader', 'innovation', 'expansion', 'dividend increase', 'merger',
            'acquisition', 'partnership', 'breakthrough', 'record high', 'all-time high'
        ]
        
        self.financial_negative = [
            'bearish', 'crash', 'plunge', 'losses', 'decline', 'recession', 'earnings miss',
            'downgrade', 'underperform', 'sell rating', 'weak performance', 'revenue decline',
            'market volatility', 'uncertainty', 'layoffs', 'dividend cut', 'bankruptcy',
            'investigation', 'lawsuit', 'regulatory issues', 'record low', 'all-time low'
        ]
        
        # Weights for different sentiment methods
        self.weights = {
            'vader': 0.4,
            'textblob': 0.3,
            'financial_keywords': 0.3
        }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Comprehensive sentiment analysis using multiple methods
        Returns scores between -1 (very negative) and 1 (very positive)
        """
        if not text or not text.strip():
            return {
                'compound_score': 0.0,
                'vader_score': 0.0,
                'textblob_score': 0.0,
                'financial_score': 0.0,
                'confidence': 0.0,
                'label': 'neutral'
            }
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
        vader_score = vader_scores['compound']
        
        # TextBlob sentiment
        blob = TextBlob(cleaned_text)
        textblob_score = blob.sentiment.polarity
        
        # Financial keyword analysis
        financial_score = self._analyze_financial_keywords(cleaned_text)
        
        # Weighted combination
        compound_score = (
            self.weights['vader'] * vader_score +
            self.weights['textblob'] * textblob_score +
            self.weights['financial_keywords'] * financial_score
        )
        
        # Calculate confidence based on agreement between methods
        confidence = self._calculate_confidence(vader_score, textblob_score, financial_score)
        
        # Determine label
        label = self._get_sentiment_label(compound_score)
        
        return {
            'compound_score': round(compound_score, 3),
            'vader_score': round(vader_score, 3),
            'textblob_score': round(textblob_score, 3),
            'financial_score': round(financial_score, 3),
            'confidence': round(confidence, 3),
            'label': label
        }
    
    def analyze_batch(self, texts: list) -> list:
        """Analyze multiple texts efficiently"""
        return [self.analyze_text(text) for text in texts]
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\-\%\$]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def _analyze_financial_keywords(self, text: str) -> float:
        """Analyze sentiment based on financial-specific keywords"""
        positive_count = sum(1 for keyword in self.financial_positive if keyword in text)
        negative_count = sum(1 for keyword in self.financial_negative if keyword in text)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return 0.0
        
        # Calculate score based on ratio
        score = (positive_count - negative_count) / total_keywords
        
        # Apply scaling factor based on keyword density
        keyword_density = total_keywords / len(text.split())
        scaling_factor = min(1.0, keyword_density * 10)  # Cap at 1.0
        
        return score * scaling_factor
    
    def _calculate_confidence(self, vader: float, textblob: float, financial: float) -> float:
        """Calculate confidence based on agreement between methods"""
        scores = [vader, textblob, financial]
        
        # Calculate standard deviation (lower = more agreement = higher confidence)
        std_dev = np.std(scores)
        
        # Convert to confidence score (0-1)
        # Lower standard deviation = higher confidence
        confidence = max(0.0, 1.0 - (std_dev / 2.0))
        
        # Boost confidence if all methods agree on direction
        signs = [1 if score > 0.1 else -1 if score < -0.1 else 0 for score in scores]
        if len(set(signs)) == 1 and signs[0] != 0:
            confidence = min(1.0, confidence + 0.2)
        
        return confidence
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert numerical score to categorical label"""
        if score >= 0.1:
            return 'positive'
        elif score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def get_market_sentiment_summary(self, articles: list) -> Dict:
        """Generate market sentiment summary from multiple articles"""
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'confidence': 0.0,
                'total_articles': 0
            }
        
        # Analyze all articles
        results = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_text(text)
            results.append(sentiment)
        
        # Calculate overall metrics
        overall_sentiment = np.mean([r['compound_score'] for r in results])
        overall_confidence = np.mean([r['confidence'] for r in results])
        
        # Count sentiment distribution
        distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        for result in results:
            distribution[result['label']] += 1
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_distribution': distribution,
            'confidence': round(overall_confidence, 3),
            'total_articles': len(articles),
            'sentiment_breakdown': results
        }

# Global analyzer instance
sentiment_analyzer = SentimentAnalyzer()
