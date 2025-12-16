"""
AI Recommendation Module for Real-Time Social Media Analytics for E-Commerce Trends
"""

from .recommendation_engine import (
    AIRecommendationEngine, 
    ai_recommendation_engine,
    TradingRecommendation,
    RecommendationAction,
    RiskLevel,
    TimeHorizon
)
from .portfolio_optimizer import (
    PortfolioOptimizer,
    portfolio_optimizer,
    PortfolioAllocation,
    RiskMetrics
)

__all__ = [
    'AIRecommendationEngine',
    'ai_recommendation_engine', 
    'TradingRecommendation',
    'RecommendationAction',
    'RiskLevel',
    'TimeHorizon',
    'PortfolioOptimizer',
    'portfolio_optimizer',
    'PortfolioAllocation',
    'RiskMetrics'
]
