"""
Advanced sentiment analysis system for financial markets
Combines multiple NLP approaches for accurate sentiment scoring
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# NLP Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Optional: Transformers for advanced analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Using basic sentiment analysis only.")

class FinancialSentimentAnalyzer:
    """
    Advanced sentiment analyzer specifically designed for financial text analysis
    """
    
    def __init__(self):
        # Initialize VADER analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformer model if available
        self.transformer_analyzer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use FinBERT for financial sentiment analysis
                self.transformer_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
            except Exception as e:
                print(f"Could not load FinBERT model: {e}")
                # Fallback to general sentiment model
                try:
                    self.transformer_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                    )
                except Exception as e2:
                    print(f"Could not load fallback model: {e2}")
        
        # Financial-specific lexicons
        self.financial_positive_terms = {
            # Market movements
            'rally', 'surge', 'soar', 'climb', 'rise', 'gain', 'jump', 'spike', 'boom',
            'bullish', 'bull market', 'uptrend', 'momentum', 'breakout', 'breakthrough',
            
            # Performance indicators
            'profit', 'earnings beat', 'revenue growth', 'strong performance', 'outperform',
            'exceed expectations', 'record high', 'all-time high', 'milestone',
            
            # Business positives
            'expansion', 'growth', 'innovation', 'partnership', 'merger', 'acquisition',
            'dividend increase', 'share buyback', 'market leader', 'competitive advantage',
            
            # Analyst sentiment
            'upgrade', 'buy rating', 'overweight', 'positive outlook', 'recommend',
            'target price increase', 'strong buy', 'conviction buy'
        }
        
        self.financial_negative_terms = {
            # Market movements
            'crash', 'plunge', 'tumble', 'fall', 'drop', 'decline', 'slide', 'sink',
            'bearish', 'bear market', 'downtrend', 'correction', 'selloff', 'rout',
            
            # Performance indicators
            'loss', 'earnings miss', 'revenue decline', 'weak performance', 'underperform',
            'disappoint', 'below expectations', 'record low', 'all-time low',
            
            # Business negatives
            'bankruptcy', 'layoffs', 'restructuring', 'debt', 'lawsuit', 'investigation',
            'regulatory issues', 'scandal', 'fraud', 'dividend cut', 'guidance cut',
            
            # Analyst sentiment
            'downgrade', 'sell rating', 'underweight', 'negative outlook', 'avoid',
            'target price cut', 'strong sell', 'reduce position'
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'very': 1.3, 'extremely': 1.5, 'highly': 1.2, 'significantly': 1.4,
            'substantially': 1.3, 'dramatically': 1.5, 'sharply': 1.4,
            'slightly': 0.7, 'somewhat': 0.8, 'moderately': 0.9,
            'barely': 0.5, 'hardly': 0.4, 'scarcely': 0.4
        }
        
        # Market context terms
        self.market_context_terms = {
            'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp', 'unemployment',
            'earnings season', 'quarterly results', 'guidance', 'outlook', 'forecast'
        }
        
        # Weights for different analysis methods
        self.method_weights = {
            'vader': 0.25,
            'textblob': 0.20,
            'financial_lexicon': 0.30,
            'transformer': 0.25 if TRANSFORMERS_AVAILABLE else 0.0
        }
        
        # Normalize weights if transformer not available
        if not TRANSFORMERS_AVAILABLE:
            total_weight = sum(self.method_weights.values())
            for key in self.method_weights:
                if key != 'transformer':
                    self.method_weights[key] = self.method_weights[key] / total_weight
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_text(self, text: str, context: str = "general") -> Dict[str, float]:
        """
        Comprehensive sentiment analysis of financial text
        
        Args:
            text: Text to analyze
            context: Context type ('news', 'social', 'earnings', 'general')
            
        Returns:
            Dictionary with sentiment scores and metadata
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Run all analysis methods
        results = {}
        
        # VADER sentiment
        results['vader'] = self._analyze_vader(cleaned_text)
        
        # TextBlob sentiment
        results['textblob'] = self._analyze_textblob(cleaned_text)
        
        # Financial lexicon analysis
        results['financial_lexicon'] = self._analyze_financial_lexicon(cleaned_text)
        
        # Transformer analysis (if available)
        if TRANSFORMERS_AVAILABLE and self.transformer_analyzer:
            results['transformer'] = self._analyze_transformer(cleaned_text)
        else:
            results['transformer'] = {'score': 0.0, 'confidence': 0.0}
        
        # Calculate weighted composite score
        composite_score = self._calculate_composite_score(results)
        
        # Calculate confidence based on agreement
        confidence = self._calculate_confidence(results)
        
        # Determine sentiment label
        sentiment_label = self._get_sentiment_label(composite_score)
        
        # Context adjustment
        adjusted_score = self._apply_context_adjustment(composite_score, context, cleaned_text)
        
        return {
            'composite_score': round(adjusted_score, 4),
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 4),
            'method_scores': {
                'vader': round(results['vader']['score'], 4),
                'textblob': round(results['textblob']['score'], 4),
                'financial_lexicon': round(results['financial_lexicon']['score'], 4),
                'transformer': round(results['transformer']['score'], 4)
            },
            'method_confidences': {
                'vader': round(results['vader']['confidence'], 4),
                'textblob': round(results['textblob']['confidence'], 4),
                'financial_lexicon': round(results['financial_lexicon']['confidence'], 4),
                'transformer': round(results['transformer']['confidence'], 4)
            },
            'text_length': len(text),
            'processed_length': len(cleaned_text),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    async def analyze_batch(self, texts: List[str], context: str = "general") -> List[Dict]:
        """
        Analyze multiple texts concurrently for better performance
        """
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [
                loop.run_in_executor(executor, self.analyze_text, text, context)
                for text in texts
            ]
            results = await asyncio.gather(*tasks)
        
        return results
    
    def analyze_market_sentiment(self, articles: List[Dict], 
                                weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Analyze overall market sentiment from multiple articles
        
        Args:
            articles: List of article dictionaries with 'title', 'description', 'source'
            weights: Optional weights for different sources
            
        Returns:
            Market sentiment summary
        """
        if not articles:
            return self._empty_market_result()
        
        # Default source weights
        if not weights:
            weights = {
                'reuters': 1.2,
                'bloomberg': 1.2,
                'wall street journal': 1.1,
                'financial times': 1.1,
                'cnbc': 1.0,
                'marketwatch': 1.0,
                'yahoo finance': 0.9,
                'default': 1.0
            }
        
        sentiment_scores = []
        confidence_scores = []
        source_distribution = {}
        
        for article in articles:
            # Combine title and description
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            # Analyze sentiment
            result = self.analyze_text(text, context='news')
            
            # Apply source weight
            source = article.get('source', '').lower()
            weight = weights.get(source, weights['default'])
            weighted_score = result['composite_score'] * weight
            
            sentiment_scores.append(weighted_score)
            confidence_scores.append(result['confidence'])
            
            # Track source distribution
            source_distribution[source] = source_distribution.get(source, 0) + 1
        
        # Calculate overall metrics
        overall_sentiment = np.mean(sentiment_scores)
        overall_confidence = np.mean(confidence_scores)
        sentiment_std = np.std(sentiment_scores)
        
        # Categorize sentiments
        positive_count = sum(1 for score in sentiment_scores if score > 0.1)
        negative_count = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        # Market mood determination
        if overall_sentiment > 0.3:
            market_mood = "Very Bullish"
        elif overall_sentiment > 0.1:
            market_mood = "Bullish"
        elif overall_sentiment > -0.1:
            market_mood = "Neutral"
        elif overall_sentiment > -0.3:
            market_mood = "Bearish"
        else:
            market_mood = "Very Bearish"
        
        return {
            'overall_sentiment': round(overall_sentiment, 4),
            'confidence': round(overall_confidence, 4),
            'volatility': round(sentiment_std, 4),
            'market_mood': market_mood,
            'sentiment_distribution': {
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count,
                'total': len(sentiment_scores)
            },
            'source_distribution': source_distribution,
            'sentiment_range': {
                'min': round(min(sentiment_scores), 4),
                'max': round(max(sentiment_scores), 4),
                'median': round(np.median(sentiment_scores), 4)
            }
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\-\%\$$$$$]', '', text)
        
        # Handle financial notation (e.g., $1.2B, 15%)
        text = re.sub(r'\$(\d+\.?\d*)[bB]', r'\1 billion dollars', text)
        text = re.sub(r'\$(\d+\.?\d*)[mM]', r'\1 million dollars', text)
        text = re.sub(r'(\d+\.?\d*)%', r'\1 percent', text)
        
        return text.strip()
    
    def _analyze_vader(self, text: str) -> Dict[str, float]:
        """VADER sentiment analysis"""
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'score': scores['compound'],
            'confidence': abs(scores['compound']),
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def _analyze_textblob(self, text: str) -> Dict[str, float]:
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert subjectivity to confidence (more subjective = more confident)
        confidence = subjectivity
        
        return {
            'score': polarity,
            'confidence': confidence,
            'subjectivity': subjectivity
        }
    
    def _analyze_financial_lexicon(self, text: str) -> Dict[str, float]:
        """Financial-specific lexicon analysis"""
        words = text.split()
        
        positive_score = 0
        negative_score = 0
        total_financial_terms = 0
        
        for i, word in enumerate(words):
            # Check for financial terms
            if word in self.financial_positive_terms:
                score = 1.0
                # Apply intensity modifier if present
                if i > 0 and words[i-1] in self.intensity_modifiers:
                    score *= self.intensity_modifiers[words[i-1]]
                positive_score += score
                total_financial_terms += 1
                
            elif word in self.financial_negative_terms:
                score = 1.0
                # Apply intensity modifier if present
                if i > 0 and words[i-1] in self.intensity_modifiers:
                    score *= self.intensity_modifiers[words[i-1]]
                negative_score += score
                total_financial_terms += 1
        
        # Calculate net sentiment
        if total_financial_terms == 0:
            return {'score': 0.0, 'confidence': 0.0}
        
        net_score = (positive_score - negative_score) / total_financial_terms
        
        # Confidence based on number of financial terms found
        confidence = min(1.0, total_financial_terms / 10.0)
        
        return {
            'score': max(-1.0, min(1.0, net_score)),
            'confidence': confidence,
            'positive_terms': positive_score,
            'negative_terms': negative_score,
            'total_terms': total_financial_terms
        }
    
    def _analyze_transformer(self, text: str) -> Dict[str, float]:
        """Transformer-based sentiment analysis"""
        if not self.transformer_analyzer:
            return {'score': 0.0, 'confidence': 0.0}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            result = self.transformer_analyzer(text)
            
            # Handle different model outputs
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            label = result.get('label', '').upper()
            confidence = result.get('score', 0.0)
            
            # Convert label to score
            if 'POSITIVE' in label or 'POS' in label:
                score = confidence
            elif 'NEGATIVE' in label or 'NEG' in label:
                score = -confidence
            else:  # NEUTRAL
                score = 0.0
            
            return {
                'score': score,
                'confidence': confidence,
                'label': label
            }
            
        except Exception as e:
            self.logger.warning(f"Transformer analysis failed: {e}")
            return {'score': 0.0, 'confidence': 0.0}
    
    def _calculate_composite_score(self, results: Dict) -> float:
        """Calculate weighted composite sentiment score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, weight in self.method_weights.items():
            if method in results and weight > 0:
                score = results[method]['score']
                confidence = results[method]['confidence']
                
                # Weight by both method weight and confidence
                effective_weight = weight * (0.5 + 0.5 * confidence)
                weighted_sum += score * effective_weight
                total_weight += effective_weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate overall confidence based on method agreement"""
        scores = []
        confidences = []
        
        for method, result in results.items():
            if method in self.method_weights and self.method_weights[method] > 0:
                scores.append(result['score'])
                confidences.append(result['confidence'])
        
        if not scores:
            return 0.0
        
        # Base confidence on individual method confidences
        avg_confidence = np.mean(confidences)
        
        # Adjust based on agreement between methods
        if len(scores) > 1:
            score_std = np.std(scores)
            # Lower standard deviation = higher agreement = higher confidence
            agreement_factor = max(0.0, 1.0 - score_std)
            avg_confidence = (avg_confidence + agreement_factor) / 2
        
        return min(1.0, avg_confidence)
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert numerical score to categorical label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _apply_context_adjustment(self, score: float, context: str, text: str) -> float:
        """Apply context-specific adjustments to sentiment score"""
        adjustment_factor = 1.0
        
        # Earnings context - amplify sentiment
        if context == 'earnings':
            if 'beat' in text or 'exceed' in text:
                adjustment_factor = 1.2
            elif 'miss' in text or 'disappoint' in text:
                adjustment_factor = 1.2
        
        # Social media context - dampen sentiment (more noise)
        elif context == 'social':
            adjustment_factor = 0.8
        
        # News context - standard weighting
        elif context == 'news':
            adjustment_factor = 1.0
        
        # Check for market context terms
        market_terms_found = sum(1 for term in self.market_context_terms if term in text)
        if market_terms_found > 0:
            # Market context increases importance
            adjustment_factor *= (1.0 + 0.1 * market_terms_found)
        
        return score * adjustment_factor
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'composite_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'method_scores': {
                'vader': 0.0,
                'textblob': 0.0,
                'financial_lexicon': 0.0,
                'transformer': 0.0
            },
            'method_confidences': {
                'vader': 0.0,
                'textblob': 0.0,
                'financial_lexicon': 0.0,
                'transformer': 0.0
            },
            'text_length': 0,
            'processed_length': 0,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def _empty_market_result(self) -> Dict:
        """Return empty market sentiment result"""
        return {
            'overall_sentiment': 0.0,
            'confidence': 0.0,
            'volatility': 0.0,
            'market_mood': 'Neutral',
            'sentiment_distribution': {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'total': 0
            },
            'source_distribution': {},
            'sentiment_range': {
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        }

# Global analyzer instance
financial_sentiment_analyzer = FinancialSentimentAnalyzer()
