from dataclasses import dataclass


@dataclass
class RiskDecision:
    severity: str
    trigger: str
    details: str


def evaluate_post_risk(sentiment_score: float, engagement: float, text: str) -> RiskDecision | None:
    lowered = (text or "").lower()
    critical_terms = {"refund", "fraud", "scam", "lawsuit", "boycott", "unsafe"}
    if any(term in lowered for term in critical_terms):
        return RiskDecision(
            severity="critical",
            trigger="brand_safety",
            details="High-risk keywords detected in social conversation.",
        )

    if sentiment_score < -0.75 and engagement > 150:
        return RiskDecision(
            severity="high",
            trigger="viral_negative_sentiment",
            details="Highly negative content with high engagement requires immediate response.",
        )

    if sentiment_score < -0.45 and engagement > 40:
        return RiskDecision(
            severity="medium",
            trigger="negative_sentiment_growth",
            details="Negative sentiment pattern indicates rising dissatisfaction.",
        )

    return None


def evaluate_product_risk(avg_sentiment: float, mention_growth: float) -> RiskDecision | None:
    if avg_sentiment < -0.40 and mention_growth > 20:
        return RiskDecision(
            severity="high",
            trigger="product_reputation_decline",
            details="Negative sentiment is increasing faster than baseline conversation volume.",
        )
    if avg_sentiment < -0.2 and mention_growth > 10:
        return RiskDecision(
            severity="medium",
            trigger="watchlist",
            details="Product sentiment trending negative. Track escalation over next windows.",
        )
    return None
