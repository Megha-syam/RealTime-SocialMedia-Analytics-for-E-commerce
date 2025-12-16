"""
AI-powered recommendation engine using Gemini LLM
Provides sophisticated product and merchandising recommendations based on comprehensive marketplace and sentiment analysis
"""

import google.generativeai as genai
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

class RecommendationAction(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class RiskLevel(Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class TimeHorizon(Enum):
    SHORT_TERM = "SHORT_TERM"  # 1-7 days
    MEDIUM_TERM = "MEDIUM_TERM"  # 1-4 weeks
    LONG_TERM = "LONG_TERM"  # 1-6 months

@dataclass
class TradingRecommendation:
    symbol: str
    action: RecommendationAction
    confidence: float  # 0-100
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_horizon: TimeHorizon
    risk_level: RiskLevel
    reasoning: str
    key_factors: List[str]
    market_conditions: str
    entry_strategy: str
    exit_strategy: str
    position_size_recommendation: str
    generated_at: datetime
    expires_at: datetime

class AIRecommendationEngine:
    """
    Advanced AI recommendation engine using Gemini LLM
    (refactored to focus on e-commerce/product merchandising recommendations;
    retains original recommendation enums/actions for backwards compatibility)
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            'gemini-pro',
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more consistent recommendations
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
        )
        
        # Risk assessment parameters
        self.risk_factors = {
            'volatility': {'weight': 0.25, 'threshold': 0.3},
            'sentiment_uncertainty': {'weight': 0.20, 'threshold': 0.4},
            'market_conditions': {'weight': 0.25, 'threshold': 0.5},
            'news_sentiment': {'weight': 0.15, 'threshold': 0.3},
            'economic_indicators': {'weight': 0.15, 'threshold': 0.4}
        }
        
        # Market condition templates
        self.market_templates = {
            'bull_market': "Strong upward trend with high investor confidence",
            'bear_market': "Declining trend with increased selling pressure",
            'sideways_market': "Range-bound trading with mixed signals",
            'volatile_market': "High volatility with rapid price movements",
            'uncertain_market': "Mixed signals requiring cautious approach"
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def generate_recommendation(self, 
                                    symbol: str,
                                    market_data: Dict,
                                    sentiment_data: Dict,
                                    news_headlines: List[str],
                                    economic_context: Dict,
                                    technical_indicators: Optional[Dict] = None) -> TradingRecommendation:
        """
        Generate comprehensive product recommendation (SKU-level)
        Note: parameter `symbol` is kept for compatibility but represents a product SKU.
        """
        try:
            # Build comprehensive analysis context (focused on product performance)
            analysis_context = self._build_analysis_context(
                symbol, market_data, sentiment_data, news_headlines, 
                economic_context, technical_indicators
            )

            # Generate AI recommendation
            ai_response = await self._query_gemini(analysis_context)

            # Parse and validate response
            recommendation = self._parse_ai_response(symbol, ai_response, sentiment_data)

            # Apply risk assessment
            recommendation = self._apply_risk_assessment(recommendation, market_data, sentiment_data)

            # Add position sizing recommendation
            recommendation.position_size_recommendation = self._calculate_position_size(
                recommendation.risk_level, recommendation.confidence
            )

            return recommendation

        except Exception as e:
            self.logger.error(f"Error generating recommendation for {symbol}: {e}")
            return self._generate_fallback_recommendation(symbol, sentiment_data)
    
    async def generate_portfolio_recommendations(self, 
                                               symbols: List[str],
                                               portfolio_context: Dict) -> List[TradingRecommendation]:
        """
        Generate recommendations for multiple symbols considering portfolio context
        """
        recommendations = []
        
        # Generate individual recommendations
        for symbol in symbols:
            try:
                # Get symbol-specific data (placeholder - would come from data sources)
                market_data = portfolio_context.get('market_data', {}).get(symbol, {})
                sentiment_data = portfolio_context.get('sentiment_data', {}).get(symbol, {})
                news_headlines = portfolio_context.get('news_headlines', {}).get(symbol, [])
                
                recommendation = await self.generate_recommendation(
                    symbol, market_data, sentiment_data, news_headlines, 
                    portfolio_context.get('economic_context', {})
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                self.logger.error(f"Error generating recommendation for {symbol}: {e}")
                continue
        
        # Apply portfolio-level adjustments
        recommendations = self._apply_portfolio_optimization(recommendations, portfolio_context)
        
        return recommendations
    
    def _build_analysis_context(self, 
                              symbol: str,
                              market_data: Dict,
                              sentiment_data: Dict,
                              news_headlines: List[str],
                              economic_context: Dict,
                              technical_indicators: Optional[Dict] = None) -> str:
        """
        Build comprehensive context for AI analysis focused on product performance
        """
        # Format news headlines
        headlines_text = "\n".join([f"- {headline}" for headline in news_headlines[:8]])

        # Format sentiment analysis
        sentiment_summary = f"""
        Overall Sentiment: {sentiment_data.get('overall_sentiment', 50):.1f}/100
        News Sentiment: {sentiment_data.get('news_sentiment', 50):.1f}/100
        Social Media Sentiment: {sentiment_data.get('social_sentiment', 50):.1f}/100
        Economic Events Sentiment: {sentiment_data.get('economic_sentiment', 50):.1f}/100
        Confidence Level: {sentiment_data.get('confidence', 0.5):.2f}
        Trend: {sentiment_data.get('trend', 'neutral')}
        Volatility: {sentiment_data.get('volatility', 0.3):.2f}
        """

        # Format market/product data
        market_summary = f"""
        Current Price: ${market_data.get('current_price', 'N/A')}
        24h Change: {market_data.get('price_change_24h', 'N/A')}%
        Volume: {market_data.get('volume', 'N/A')}
        Market Cap: ${market_data.get('market_cap', 'N/A')}
        Additional Metrics: {market_data.get('additional_metrics', 'N/A')}
        """

        # Format economic/context
        economic_summary = f"""
        Marketplace Mood: {economic_context.get('market_mood', 'Neutral')}
        Key Events: {', '.join(economic_context.get('upcoming_events', [])[:3])}
        Supply Chain Notes: {economic_context.get('supply_chain_notes', 'None')}
        """

        # Technical indicators (if available)
        technical_summary = ""
        if technical_indicators:
            technical_summary = f"""
        RSI: {technical_indicators.get('rsi', 'N/A')}
        MACD: {technical_indicators.get('macd', 'N/A')}
        Moving Averages: {technical_indicators.get('moving_averages', 'N/A')}
        Support/Resistance: {technical_indicators.get('support_resistance', 'N/A')}
        """

        context = f"""
        You are an experienced e-commerce/product merchandising analyst. Analyze the following comprehensive data for {symbol} (SKU) and provide a clear recommendation focused on merchandising, pricing, promotion, or inventory actions.

        PRODUCT DATA / MARKETPLACE METRICS:
        {market_summary}

        SENTIMENT ANALYSIS:
        {sentiment_summary}

        RECENT HEADLINES / REVIEWS / MENTIONS:
        {headlines_text}

        ECONOMIC / MARKETPLACE CONTEXT:
        {economic_summary}

        {f"TECHNICAL INDICATORS:{technical_summary}" if technical_indicators else ""}

        ANALYSIS REQUIREMENTS:
        1. Provide a clear recommendation: PROMOTE, DISCOUNT, or HOLD (map to BUY/SELL/HOLD for compatibility)
        2. Give a confidence percentage (60-95%)
        3. Suggest pricing or promotion guidance (e.g., % discount, promotional channel)
        4. Identify an action time horizon: SHORT_TERM, MEDIUM_TERM, or LONG_TERM
        5. Assess risk level: VERY_LOW, LOW, MEDIUM, HIGH, or VERY_HIGH
        6. Provide detailed reasoning (2-4 sentences)
        7. List 3-5 key factors influencing your decision (reviews, competitors, inventory)
        8. Describe marketplace conditions affecting the product
        9. Suggest operational actions (inventory, supplier, promotional channels)

        IMPORTANT GUIDELINES:
        - Be pragmatic when data is limited; prefer conservative operational actions
        - Consider sentiment, review quality, and competitor pricing
        - Provide actionable, specific merchandising advice
        - Acknowledge limitations and risks

        Format your response as a structured analysis with clear sections.
        """

        return context
    
    async def _query_gemini(self, context: str) -> str:
        """
        Query Gemini model with retry logic
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(context)
                return response.text
                
            except Exception as e:
                self.logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e
    
    def _parse_ai_response(self, symbol: str, ai_text: str, sentiment_data: Dict) -> TradingRecommendation:
        """
        Parse AI response into structured recommendation
        """
        # Extract recommendation action
        action = self._extract_action(ai_text)
        
        # Extract confidence
        confidence = self._extract_confidence(ai_text)
        
        # Extract target price and stop loss
        target_price, stop_loss = self._extract_price_levels(ai_text)
        
        # Extract time horizon
        time_horizon = self._extract_time_horizon(ai_text)
        
        # Extract risk level
        risk_level = self._extract_risk_level(ai_text)
        
        # Extract reasoning and key factors
        reasoning = self._extract_reasoning(ai_text)
        key_factors = self._extract_key_factors(ai_text)
        
        # Extract market conditions
        market_conditions = self._extract_market_conditions(ai_text)
        
        # Extract strategies
        entry_strategy = self._extract_entry_strategy(ai_text)
        exit_strategy = self._extract_exit_strategy(ai_text)
        
        # Set expiration (recommendations expire after 24 hours)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        return TradingRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon=time_horizon,
            risk_level=risk_level,
            reasoning=reasoning,
            key_factors=key_factors,
            market_conditions=market_conditions,
            entry_strategy=entry_strategy,
            exit_strategy=exit_strategy,
            position_size_recommendation="",  # Will be filled later
            generated_at=datetime.utcnow(),
            expires_at=expires_at
        )
    
    def _extract_action(self, text: str) -> RecommendationAction:
        """Extract recommendation action from AI response"""
        text_upper = text.upper()
        
        if "STRONG BUY" in text_upper or "STRONG_BUY" in text_upper:
            return RecommendationAction.STRONG_BUY
        elif "STRONG SELL" in text_upper or "STRONG_SELL" in text_upper:
            return RecommendationAction.STRONG_SELL
        elif "BUY" in text_upper:
            return RecommendationAction.BUY
        elif "SELL" in text_upper:
            return RecommendationAction.SELL
        else:
            return RecommendationAction.HOLD
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence percentage from AI response"""
        # Look for patterns like "85% confidence", "confidence: 75%", etc.
        patterns = [
            r'confidence[:\s]+(\d+)%',
            r'(\d+)%\s+confidence',
            r'confidence[:\s]+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # Default confidence based on text sentiment
        return 75.0
    
    def _extract_price_levels(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract target price and stop loss from AI response"""
        target_price = None
        stop_loss = None
        
        # Look for target price patterns
        target_patterns = [
            r'target[:\s]+\$?(\d+\.?\d*)',
            r'price target[:\s]+\$?(\d+\.?\d*)',
            r'target price[:\s]+\$?(\d+\.?\d*)',
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                target_price = float(match.group(1))
                break
        
        # Look for stop loss patterns
        stop_patterns = [
            r'stop loss[:\s]+\$?(\d+\.?\d*)',
            r'stop[:\s]+\$?(\d+\.?\d*)',
            r'stop-loss[:\s]+\$?(\d+\.?\d*)',
        ]
        
        for pattern in stop_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                stop_loss = float(match.group(1))
                break
        
        return target_price, stop_loss
    
    def _extract_time_horizon(self, text: str) -> TimeHorizon:
        """Extract time horizon from AI response"""
        text_upper = text.upper()
        
        if any(term in text_upper for term in ["SHORT TERM", "SHORT-TERM", "1-7 DAYS", "DAYS"]):
            return TimeHorizon.SHORT_TERM
        elif any(term in text_upper for term in ["LONG TERM", "LONG-TERM", "MONTHS", "1-6 MONTHS"]):
            return TimeHorizon.LONG_TERM
        else:
            return TimeHorizon.MEDIUM_TERM
    
    def _extract_risk_level(self, text: str) -> RiskLevel:
        """Extract risk level from AI response"""
        text_upper = text.upper()
        
        if "VERY HIGH" in text_upper:
            return RiskLevel.VERY_HIGH
        elif "VERY LOW" in text_upper:
            return RiskLevel.VERY_LOW
        elif "HIGH" in text_upper:
            return RiskLevel.HIGH
        elif "LOW" in text_upper:
            return RiskLevel.LOW
        else:
            return RiskLevel.MEDIUM
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from AI response"""
        # Look for reasoning section
        reasoning_patterns = [
            r'reasoning[:\s]+(.*?)(?=key factors|market conditions|entry strategy|\n\n|$)',
            r'analysis[:\s]+(.*?)(?=key factors|market conditions|entry strategy|\n\n|$)',
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:500]  # Limit length
        
        # Fallback: take first few sentences
        sentences = text.split('.')[:3]
        return '. '.join(sentences) + '.'
    
    def _extract_key_factors(self, text: str) -> List[str]:
        """Extract key factors from AI response"""
        factors = []
        
        # Look for bullet points or numbered lists
        factor_patterns = [
            r'[•\-\*]\s*([^\n]+)',
            r'\d+\.\s*([^\n]+)',
        ]
        
        for pattern in factor_patterns:
            matches = re.findall(pattern, text)
            factors.extend([match.strip() for match in matches])
        
        # Limit to 5 factors
        return factors[:5] if factors else ["Market sentiment analysis", "Technical indicators", "Economic conditions"]
    
    def _extract_market_conditions(self, text: str) -> str:
        """Extract market conditions description"""
        conditions_patterns = [
            r'market conditions[:\s]+(.*?)(?=entry strategy|exit strategy|\n\n|$)',
            r'current market[:\s]+(.*?)(?=entry strategy|exit strategy|\n\n|$)',
        ]
        
        for pattern in conditions_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:200]
        
        return "Mixed market conditions with moderate volatility"
    
    def _extract_entry_strategy(self, text: str) -> str:
        """Extract entry strategy from AI response"""
        entry_patterns = [
            r'entry strategy[:\s]+(.*?)(?=exit strategy|position sizing|\n\n|$)',
            r'entry[:\s]+(.*?)(?=exit strategy|position sizing|\n\n|$)',
        ]
        
        for pattern in entry_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:200]
        
        return "Consider dollar-cost averaging for entry"
    
    def _extract_exit_strategy(self, text: str) -> str:
        """Extract exit strategy from AI response"""
        exit_patterns = [
            r'exit strategy[:\s]+(.*?)(?=position sizing|\n\n|$)',
            r'exit[:\s]+(.*?)(?=position sizing|\n\n|$)',
        ]
        
        for pattern in exit_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:200]
        
        return "Use trailing stop loss to protect profits"
    
    def _apply_risk_assessment(self, 
                             recommendation: TradingRecommendation,
                             market_data: Dict,
                             sentiment_data: Dict) -> TradingRecommendation:
        """
        Apply additional risk assessment and adjust recommendation if needed
        """
        risk_score = 0.0
        
        # Assess volatility risk
        volatility = sentiment_data.get('volatility', 0.3)
        if volatility > self.risk_factors['volatility']['threshold']:
            risk_score += self.risk_factors['volatility']['weight']
        
        # Assess sentiment uncertainty
        confidence = sentiment_data.get('confidence', 0.5)
        if confidence < self.risk_factors['sentiment_uncertainty']['threshold']:
            risk_score += self.risk_factors['sentiment_uncertainty']['weight']
        
        # Assess market conditions
        market_sentiment = sentiment_data.get('overall_sentiment', 50)
        if abs(market_sentiment - 50) < 10:  # Neutral/uncertain sentiment
            risk_score += self.risk_factors['market_conditions']['weight']
        
        # Adjust recommendation based on risk score
        if risk_score > 0.6:  # High risk
            # Downgrade aggressive recommendations
            if recommendation.action == RecommendationAction.STRONG_BUY:
                recommendation.action = RecommendationAction.BUY
            elif recommendation.action == RecommendationAction.STRONG_SELL:
                recommendation.action = RecommendationAction.SELL
            
            # Reduce confidence
            recommendation.confidence = max(60, recommendation.confidence - 10)
            
            # Increase risk level
            if recommendation.risk_level == RiskLevel.LOW:
                recommendation.risk_level = RiskLevel.MEDIUM
            elif recommendation.risk_level == RiskLevel.MEDIUM:
                recommendation.risk_level = RiskLevel.HIGH
        
        return recommendation
    
    def _calculate_position_size(self, risk_level: RiskLevel, confidence: float) -> str:
        """
        Calculate position size recommendation based on risk and confidence
        """
        base_size = {
            RiskLevel.VERY_LOW: "5-10% of portfolio",
            RiskLevel.LOW: "3-7% of portfolio", 
            RiskLevel.MEDIUM: "2-5% of portfolio",
            RiskLevel.HIGH: "1-3% of portfolio",
            RiskLevel.VERY_HIGH: "0.5-2% of portfolio"
        }
        
        size_recommendation = base_size.get(risk_level, "2-5% of portfolio")
        
        # Adjust based on confidence
        if confidence > 85:
            size_recommendation += " (consider higher allocation due to high confidence)"
        elif confidence < 70:
            size_recommendation += " (consider lower allocation due to uncertainty)"
        
        return size_recommendation
    
    def _apply_portfolio_optimization(self, 
                                    recommendations: List[TradingRecommendation],
                                    portfolio_context: Dict) -> List[TradingRecommendation]:
        """
        Apply portfolio-level optimization to recommendations
        """
        # Count recommendations by action
        action_counts = {}
        for rec in recommendations:
            action_counts[rec.action] = action_counts.get(rec.action, 0) + 1
        
        # If too many BUY recommendations, adjust some to HOLD
        total_buys = action_counts.get(RecommendationAction.BUY, 0) + action_counts.get(RecommendationAction.STRONG_BUY, 0)
        
        if total_buys > len(recommendations) * 0.6:  # More than 60% buy recommendations
            # Sort by confidence and adjust lower confidence ones
            buy_recs = [r for r in recommendations if r.action in [RecommendationAction.BUY, RecommendationAction.STRONG_BUY]]
            buy_recs.sort(key=lambda x: x.confidence)
            
            # Adjust bottom 30% to HOLD
            adjust_count = int(len(buy_recs) * 0.3)
            for i in range(adjust_count):
                buy_recs[i].action = RecommendationAction.HOLD
                buy_recs[i].reasoning += " (Adjusted to HOLD for portfolio diversification)"
        
        return recommendations
    
    def _generate_fallback_recommendation(self, symbol: str, sentiment_data: Dict) -> TradingRecommendation:
        """
        Generate fallback recommendation when AI is unavailable
        """
        overall_sentiment = sentiment_data.get('overall_sentiment', 50)
        
        if overall_sentiment > 70:
            action = RecommendationAction.BUY
            confidence = 65
        elif overall_sentiment < 30:
            action = RecommendationAction.SELL
            confidence = 65
        else:
            action = RecommendationAction.HOLD
            confidence = 60
        
        return TradingRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            target_price=None,
            stop_loss=None,
            time_horizon=TimeHorizon.MEDIUM_TERM,
            risk_level=RiskLevel.MEDIUM,
            reasoning="AI recommendation service temporarily unavailable. Recommendation based on sentiment analysis only.",
            key_factors=["Sentiment analysis", "Market conditions", "Risk management"],
            market_conditions="Unable to assess current market conditions",
            entry_strategy="Wait for AI service to resume for detailed entry strategy",
            exit_strategy="Use standard risk management practices",
            position_size_recommendation="1-3% of portfolio (conservative due to limited analysis)",
            generated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )

# Global recommendation engine instance
ai_recommendation_engine = AIRecommendationEngine("AIzaSyAYPTIR62vJ7dEyGPKN7ykzBSsdWcB7Vu8")
