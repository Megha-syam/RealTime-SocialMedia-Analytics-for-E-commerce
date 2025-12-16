"""
API clients for external data sources
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

class NewsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsdata.io/api/1"
    
    async def get_ecommerce_news(self, 
                               category: str = "business",
                               country: str = "us",
                               language: str = "en",
                               size: int = 50) -> List[Dict]:
        """Fetch e-commerce and retail related news from NewsData API"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "apikey": self.api_key,
                "category": category,
                "country": country,
                "language": language,
                "size": size,
             # Focus query on e-commerce, retail, marketplaces and product trends
             "q": "ecommerce OR retail OR marketplace OR product OR " + \
                 "shopping OR reviews OR rating OR "
            }
            
            try:
                response = await client.get(f"{self.base_url}/news", params=params)
                response.raise_for_status()
                
                data = response.json()
                return data.get("results", [])
                
            except httpx.HTTPError as e:
                print(f"Error fetching news: {e}")
                return []
    
    async def search_sku_news(self, sku: str, days: int = 7) -> List[Dict]:
        """Search for news specific to a product SKU or product name"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "apikey": self.api_key,
                "q": f"{sku} OR {sku.lower()}",
                "language": "en",
                "size": 20
            }
            
            try:
                response = await client.get(f"{self.base_url}/news", params=params)
                response.raise_for_status()
                
                data = response.json()
                return data.get("results", [])
                
            except httpx.HTTPError as e:
                print(f"Error searching symbol news: {e}")
                return []

class TradingEconomicsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
    
    async def get_economic_calendar(self, 
                                  country: Optional[str] = None,
                                  days: int = 7) -> List[Dict]:
        """Get economic calendar events"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.base_url}/calendar"
            params = {"c": self.api_key}
            
            if country:
                params["country"] = country
            
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Filter recent events
                recent_events = []
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for event in data:
                    event_date_str = event.get("Date", "")
                    if event_date_str:
                        try:
                            event_date = datetime.fromisoformat(event_date_str.replace("T", " "))
                            if event_date >= cutoff_date:
                                recent_events.append(event)
                        except ValueError:
                            continue
                
                return recent_events
                
            except httpx.HTTPError as e:
                print(f"Error fetching economic calendar: {e}")
                return []
    
    async def get_market_indicators(self, country: str = "united states") -> List[Dict]:
        """Get macroeconomic indicators (kept for context)"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.base_url}/indicators"
            params = {
                "c": self.api_key,
                "country": country
            }
            
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                return response.json()
                
            except httpx.HTTPError as e:
                print(f"Error fetching market indicators: {e}")
                return []

class GeminiAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._configure_client()
    
    def _configure_client(self):
        """Configure Gemini AI client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        except ImportError:
            print("Google Generative AI library not installed")
            self.model = None
    
    async def generate_product_recommendation(self, 
                                            sku: str,
                                            sentiment_data: Dict,
                                            news_headlines: List[str],
                                            economic_context: str = "") -> Dict:
        """Generate AI-powered e-commerce recommendation (promote/discount/restock)"""
        if not self.model:
            return self._fallback_recommendation(sku)
        
        try:
            # Prepare comprehensive context
            context = self._build_analysis_context(
                sku, sentiment_data, news_headlines, economic_context
            )
            
            # Generate recommendation
            response = self.model.generate_content(context)

            # Parse and structure the response
            return self._parse_ai_response(sku, response.text, sentiment_data)

        except Exception as e:
            print(f"Error generating AI recommendation: {e}")
            return self._fallback_recommendation(sku)
    
    def _build_analysis_context(self, 
                              sku: str,
                              sentiment_data: Dict,
                              news_headlines: List[str],
                              economic_context: str) -> str:
        """Build comprehensive context for AI analysis focused on product performance"""
        headlines_text = "\n".join([f"- {headline}" for headline in news_headlines[:5]])
        
        context = f"""
        As an experienced e-commerce analyst, analyze the following product performance data for {sku} and provide a recommendation focused on promotion, pricing, or inventory actions.

        PRODUCT PERFORMANCE SUMMARY:
        - Overall Sentiment Score: {sentiment_data.get('overall_sentiment', 0):.1f}/100
        - News/Article Sentiment: {sentiment_data.get('news_sentiment', 0):.1f}/100
        - Social Media Sentiment: {sentiment_data.get('social_sentiment', 0):.1f}/100
        - Economic/Market Context Impact: {sentiment_data.get('economic_sentiment', 0):.1f}/100
        - Confidence Level: {sentiment_data.get('confidence', 0):.1f}

        RECENT HEADLINES / REVIEWS / MENTIONS:
        {headlines_text}

        ECONOMIC CONTEXT / MARKETPLACE NOTES:
        {economic_context}

        INSTRUCTIONS:
        1. Provide a clear recommendation: PROMOTE, DISCOUNT, or HOLD (map to BUY/SELL/HOLD for compatibility)
        2. Give a confidence percentage (60-95%)
        3. Explain your reasoning in 2-3 sentences
        4. Identify key risks (supply, demand, reviews, competitors)
        5. Suggest an action time horizon (Short/Medium/Long)

        Format your response as:
        RECOMMENDATION: [PROMOTE/DISCOUNT/HOLD]
        CONFIDENCE: [XX]%
        REASONING: [Your analysis]
        RISKS: [Key risks to consider]
        TIME_HORIZON: [Short/Medium/Long]
        """

        return context
    
    def _parse_ai_response(self, sku: str, ai_text: str, sentiment_data: Dict) -> Dict:
        """Parse AI response into structured format"""
        # Extract recommendation
        action = "HOLD"
        text_up = ai_text.upper()
        if "PROMOTE" in text_up or "RECOMMENDATION: PROMOTE" in text_up or "RECOMMENDATION: BUY" in text_up:
            action = "BUY"  # map PROMOTE -> BUY for compatibility
        elif "DISCOUNT" in text_up or "RECOMMENDATION: DISCOUNT" in text_up or "RECOMMENDATION: SELL" in text_up:
            action = "SELL"  # map DISCOUNT -> SELL for compatibility
        
        # Extract confidence
        confidence = 75  # Default
        try:
            import re
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)%', ai_text)
            if confidence_match:
                confidence = int(confidence_match.group(1))
        except:
            pass
        
        # Extract reasoning
        reasoning_start = ai_text.find("REASONING:")
        reasoning_end = ai_text.find("RISKS:")
        if reasoning_start != -1:
            reasoning = ai_text[reasoning_start + 10:reasoning_end].strip()
        else:
            reasoning = "AI analysis completed based on current market sentiment and news."
        
        # Determine risk level
        risk_level = "Medium"
        if confidence >= 85:
            risk_level = "Low"
        elif confidence <= 70:
            risk_level = "High"
        
        return {
            "symbol": sku,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning[:300],  # Limit length
            "risk_level": risk_level,
            "generated_at": datetime.now().isoformat()
        }
    
    def _fallback_recommendation(self, sku: str) -> Dict:
        """Fallback recommendation when AI is unavailable"""
        return {
            "symbol": sku,
            "action": "HOLD",
            "confidence": 50,
            "reasoning": "AI recommendation service temporarily unavailable. Please consult additional sources before making product decisions.",
            "risk_level": "High",
            "generated_at": datetime.now().isoformat()
        }

# Initialize clients with API keys
news_client = NewsAPIClient("pub_f6cb8108c3dd449299f965cd7c131702")
economics_client = TradingEconomicsClient("b4a2da19-2dec-4a25-a97e-290d452a91d4")
ai_client = GeminiAIClient("AIzaSyCnvqMspISmF_IAFkTO89czQkRxoKHLQzY")
