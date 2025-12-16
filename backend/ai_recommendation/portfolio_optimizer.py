"""
Portfolio optimization and risk management for AI recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .recommendation_engine import TradingRecommendation, RecommendationAction, RiskLevel

@dataclass
class PortfolioAllocation:
    symbol: str
    current_weight: float
    recommended_weight: float
    action: str  # INCREASE, DECREASE, MAINTAIN
    rationale: str

@dataclass
class RiskMetrics:
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    beta: float
    correlation_risk: float

class PortfolioOptimizer:
    """
    Portfolio optimization engine for AI recommendations
    """
    
    def __init__(self):
        self.max_single_position = 0.15  # 15% max per position
        self.max_sector_allocation = 0.30  # 30% max per sector
        self.target_portfolio_volatility = 0.20  # 20% annual volatility target
        
        # Risk budgets by asset class
        self.risk_budgets = {
            'large_cap': 0.40,
            'mid_cap': 0.25,
            'small_cap': 0.15,
            'international': 0.20
        }
        
        # Sector limits
        self.sector_limits = {
            'technology': 0.25,
            'healthcare': 0.20,
            'financial': 0.20,
            'consumer': 0.15,
            'industrial': 0.15,
            'energy': 0.10,
            'utilities': 0.10,
            'materials': 0.10
        }
    
    def optimize_portfolio(self, 
                          recommendations: List[TradingRecommendation],
                          current_portfolio: Dict[str, float],
                          market_data: Dict,
                          risk_tolerance: str = "moderate") -> Dict:
        """
        Optimize portfolio allocation based on AI recommendations
        """
        # Calculate recommended allocations
        allocations = self._calculate_allocations(recommendations, current_portfolio, risk_tolerance)
        
        # Apply risk constraints
        allocations = self._apply_risk_constraints(allocations, market_data)
        
        # Calculate portfolio metrics
        risk_metrics = self._calculate_risk_metrics(allocations, market_data)
        
        # Generate rebalancing actions
        rebalancing_actions = self._generate_rebalancing_actions(
            allocations, current_portfolio
        )
        
        return {
            'allocations': allocations,
            'risk_metrics': risk_metrics,
            'rebalancing_actions': rebalancing_actions,
            'optimization_timestamp': datetime.utcnow(),
            'risk_tolerance': risk_tolerance
        }
    
    def assess_portfolio_risk(self, 
                            portfolio: Dict[str, float],
                            market_data: Dict) -> RiskMetrics:
        """
        Assess current portfolio risk metrics
        """
        return self._calculate_risk_metrics(portfolio, market_data)
    
    def generate_diversification_recommendations(self, 
                                               current_portfolio: Dict[str, float],
                                               market_data: Dict) -> List[str]:
        """
        Generate recommendations to improve portfolio diversification
        """
        recommendations = []
        
        # Check concentration risk
        sorted_positions = sorted(current_portfolio.items(), key=lambda x: x[1], reverse=True)
        
        # Check for over-concentration
        if sorted_positions and sorted_positions[0][1] > self.max_single_position:
            recommendations.append(
                f"Reduce concentration in {sorted_positions[0][0]} "
                f"({sorted_positions[0][1]:.1%} of portfolio)"
            )
        
        # Check top 3 positions
        top_3_weight = sum(weight for _, weight in sorted_positions[:3])
        if top_3_weight > 0.50:
            recommendations.append(
                "Top 3 positions represent over 50% of portfolio - consider diversification"
            )
        
        # Check sector diversification
        sector_allocations = self._calculate_sector_allocations(current_portfolio, market_data)
        for sector, allocation in sector_allocations.items():
            if allocation > self.sector_limits.get(sector, 0.20):
                recommendations.append(
                    f"Over-allocated to {sector} sector ({allocation:.1%})"
                )
        
        # Check for missing asset classes
        if len(current_portfolio) < 8:
            recommendations.append(
                "Consider adding more positions for better diversification"
            )
        
        return recommendations
    
    def _calculate_allocations(self, 
                             recommendations: List[TradingRecommendation],
                             current_portfolio: Dict[str, float],
                             risk_tolerance: str) -> Dict[str, PortfolioAllocation]:
        """
        Calculate optimal portfolio allocations
        """
        allocations = {}
        
        # Risk tolerance multipliers
        risk_multipliers = {
            'conservative': 0.7,
            'moderate': 1.0,
            'aggressive': 1.3
        }
        
        risk_multiplier = risk_multipliers.get(risk_tolerance, 1.0)
        
        # Calculate base allocations from recommendations
        total_confidence_score = 0
        recommendation_scores = {}
        
        for rec in recommendations:
            # Calculate allocation score based on action, confidence, and risk
            action_multiplier = {
                RecommendationAction.STRONG_BUY: 1.5,
                RecommendationAction.BUY: 1.0,
                RecommendationAction.HOLD: 0.5,
                RecommendationAction.SELL: 0.0,
                RecommendationAction.STRONG_SELL: 0.0
            }.get(rec.action, 0.5)
            
            risk_adjustment = {
                RiskLevel.VERY_LOW: 1.2,
                RiskLevel.LOW: 1.1,
                RiskLevel.MEDIUM: 1.0,
                RiskLevel.HIGH: 0.8,
                RiskLevel.VERY_HIGH: 0.6
            }.get(rec.risk_level, 1.0)
            
            score = (rec.confidence / 100) * action_multiplier * risk_adjustment * risk_multiplier
            recommendation_scores[rec.symbol] = score
            total_confidence_score += score
        
        # Normalize allocations
        if total_confidence_score > 0:
            for symbol, score in recommendation_scores.items():
                base_allocation = score / total_confidence_score
                current_weight = current_portfolio.get(symbol, 0.0)
                
                # Determine action
                if base_allocation > current_weight + 0.02:  # 2% threshold
                    action = "INCREASE"
                elif base_allocation < current_weight - 0.02:
                    action = "DECREASE"
                else:
                    action = "MAINTAIN"
                
                allocations[symbol] = PortfolioAllocation(
                    symbol=symbol,
                    current_weight=current_weight,
                    recommended_weight=min(base_allocation, self.max_single_position),
                    action=action,
                    rationale=f"Based on AI recommendation with {score:.2f} allocation score"
                )
        
        return allocations
    
    def _apply_risk_constraints(self, 
                              allocations: Dict[str, PortfolioAllocation],
                              market_data: Dict) -> Dict[str, PortfolioAllocation]:
        """
        Apply risk constraints to allocations
        """
        # Apply single position limits
        for symbol, allocation in allocations.items():
            if allocation.recommended_weight > self.max_single_position:
                allocation.recommended_weight = self.max_single_position
                allocation.rationale += f" (Capped at {self.max_single_position:.1%} position limit)"
        
        # Apply sector limits
        sector_allocations = {}
        for symbol, allocation in allocations.items():
            sector = market_data.get('sectors', {}).get(symbol, 'unknown')
            sector_allocations[sector] = sector_allocations.get(sector, 0) + allocation.recommended_weight
        
        # Adjust if sector limits exceeded
        for sector, total_allocation in sector_allocations.items():
            sector_limit = self.sector_limits.get(sector, 0.20)
            if total_allocation > sector_limit:
                # Proportionally reduce allocations in this sector
                reduction_factor = sector_limit / total_allocation
                for symbol, allocation in allocations.items():
                    symbol_sector = market_data.get('sectors', {}).get(symbol, 'unknown')
                    if symbol_sector == sector:
                        allocation.recommended_weight *= reduction_factor
                        allocation.rationale += f" (Adjusted for {sector} sector limit)"
        
        # Normalize to ensure total doesn't exceed 100%
        total_allocation = sum(alloc.recommended_weight for alloc in allocations.values())
        if total_allocation > 1.0:
            for allocation in allocations.values():
                allocation.recommended_weight /= total_allocation
                allocation.rationale += " (Normalized to 100% portfolio)"
        
        return allocations
    
    def _calculate_risk_metrics(self, 
                              portfolio: Dict,
                              market_data: Dict) -> RiskMetrics:
        """
        Calculate portfolio risk metrics
        """
        # Placeholder implementation - in production, use historical data
        # and proper risk calculation methods
        
        # Estimate portfolio volatility (simplified)
        portfolio_volatility = 0.15  # 15% annual volatility estimate
        
        # Estimate Sharpe ratio
        risk_free_rate = 0.03  # 3% risk-free rate
        expected_return = 0.08  # 8% expected return
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility
        
        # Estimate other metrics
        max_drawdown = 0.20  # 20% max drawdown estimate
        var_95 = portfolio_volatility * 1.65  # 95% VaR approximation
        beta = 1.0  # Market beta
        correlation_risk = 0.3  # Correlation risk estimate
        
        return RiskMetrics(
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            beta=beta,
            correlation_risk=correlation_risk
        )
    
    def _calculate_sector_allocations(self, 
                                    portfolio: Dict[str, float],
                                    market_data: Dict) -> Dict[str, float]:
        """
        Calculate current sector allocations
        """
        sector_allocations = {}
        
        for symbol, weight in portfolio.items():
            sector = market_data.get('sectors', {}).get(symbol, 'unknown')
            sector_allocations[sector] = sector_allocations.get(sector, 0) + weight
        
        return sector_allocations
    
    def _generate_rebalancing_actions(self, 
                                    allocations: Dict[str, PortfolioAllocation],
                                    current_portfolio: Dict[str, float]) -> List[Dict]:
        """
        Generate specific rebalancing actions
        """
        actions = []
        
        for symbol, allocation in allocations.items():
            current_weight = current_portfolio.get(symbol, 0.0)
            target_weight = allocation.recommended_weight
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # 1% threshold for action
                action_type = "BUY" if weight_diff > 0 else "SELL"
                
                actions.append({
                    'symbol': symbol,
                    'action': action_type,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_change': weight_diff,
                    'priority': abs(weight_diff),  # Higher difference = higher priority
                    'rationale': allocation.rationale
                })
        
        # Sort by priority (largest changes first)
        actions.sort(key=lambda x: x['priority'], reverse=True)
        
        return actions

# Global portfolio optimizer instance
portfolio_optimizer = PortfolioOptimizer()
