import { NextResponse } from "next/server"

// Helper to safely extract message from unknown errors for logging
function getErrorMessage(err: unknown): string {
  if (typeof err === "string") return err
  if (err instanceof Error) return err.message
  try {
    return JSON.stringify(err)
  } catch {
    return String(err)
  }
}

// Product data service to fetch basic product information (placeholder for e-commerce integrations)
async function fetchMarketData(symbol: string) {
  try {
  // Placeholder: using a free market API for demo purposes. Replace with product catalog or sales API.
    const response = await fetch(
      `https://api.polygon.io/v2/aggs/ticker/${symbol}/prev?adjusted=true&apikey=demo`, // Using demo key
      { next: { revalidate: 300 } }, // Cache for 5 minutes
    )

    if (response.ok) {
      const data = await response.json()
      return data.results?.[0] || null
    }
  } catch (error) {
    console.log(`[v0] Market data unavailable for ${symbol}:`, error)
  }
  return null
}

// News sentiment analysis
async function fetchNewsData(symbol: string) {
  try {
  // Using NewsAPI for real news data; in e-commerce this can represent product mentions, reviews, or social posts
    const response = await fetch(
      `https://newsapi.org/v2/everything?q=${symbol}&sortBy=publishedAt&pageSize=5&apiKey=demo`, // Using demo key
      { next: { revalidate: 600 } }, // Cache for 10 minutes
    )

    if (response.ok) {
      const data = await response.json()
      return data.articles || []
    }
  } catch (error) {
    console.log(`[v0] News data unavailable for ${symbol}:`, error)
  }
  return []
}

// Simple sentiment analysis function
function analyzeSentiment(text: string): { score: number; label: string } {
  const positiveWords = [
    "buy",
    "bull",
    "bullish",
    "growth",
    "profit",
    "gain",
    "rise",
    "up",
    "strong",
    "good",
    "positive",
    "excellent",
    "outperform",
  ]
  const negativeWords = [
    "sell",
    "bear",
    "bearish",
    "loss",
    "decline",
    "fall",
    "down",
    "weak",
    "bad",
    "negative",
    "poor",
    "underperform",
  ]

  const words = text.toLowerCase().split(/\W+/)
  const positiveCount = words.filter((word) => positiveWords.includes(word)).length
  const negativeCount = words.filter((word) => negativeWords.includes(word)).length

  const score = (positiveCount - negativeCount) / Math.max(words.length / 10, 1)
  const normalizedScore = Math.max(-1, Math.min(1, score))

  let label = "neutral"
  if (normalizedScore > 0.1) label = "positive"
  else if (normalizedScore < -0.1) label = "negative"

  return { score: normalizedScore, label }
}

function generateFallbackRecommendation(symbol: string, marketData: any, newsData: any[]): any {
  const sentiment =
    newsData.length > 0
      ? analyzeSentiment(newsData.map((article) => `${article.title} ${article.description || ""}`).join(" "))
      : { score: 0, label: "neutral" }

  // Simple rule-based recommendation logic
  let action = "HOLD"
  let confidence = 60
  let reasoning = "Based on basic sentiment analysis and market indicators"

  if (sentiment.score > 0.3) {
    action = "BUY"
    confidence = 70
    reasoning = "Positive news sentiment detected. Market indicators suggest potential upward movement."
  } else if (sentiment.score < -0.3) {
    action = "SELL"
    confidence = 65
    reasoning = "Negative news sentiment detected. Consider reducing position or taking profits."
  } else {
    reasoning = "Mixed or neutral sentiment. Recommend holding current position and monitoring developments."
  }

  // Adjust confidence based on data availability
  if (!marketData) confidence -= 10
  if (newsData.length === 0) confidence -= 15

  return {
    symbol,
    action,
    confidence: Math.max(50, Math.min(85, confidence)),
    reasoning,
    target_price: marketData?.c ? Math.round(marketData.c * (sentiment.score > 0 ? 1.05 : 0.95) * 100) / 100 : null,
    stop_loss: marketData?.c ? Math.round(marketData.c * (sentiment.score > 0 ? 0.92 : 1.08) * 100) / 100 : null,
    risk_level: Math.abs(sentiment.score) > 0.5 ? "High" : "Medium",
    data_source: "FALLBACK_ANALYSIS",
    timestamp: new Date().toISOString(),
    market_data_available: !!marketData,
    news_sentiment: sentiment.label,
    news_count: newsData.length,
  }
}

let apiCallCount = 0
let lastResetTime = Date.now()
const DAILY_QUOTA_LIMIT = 45 // Leave some buffer from the 50 limit
const RESET_INTERVAL = 24 * 60 * 60 * 1000 // 24 hours

function checkQuotaAvailable(): boolean {
  const now = Date.now()

  // Reset counter if 24 hours have passed
  if (now - lastResetTime > RESET_INTERVAL) {
    apiCallCount = 0
    lastResetTime = now
  }

  return apiCallCount < DAILY_QUOTA_LIMIT
}

function incrementQuotaUsage() {
  apiCallCount++
  console.log(`[v0] API quota usage: ${apiCallCount}/${DAILY_QUOTA_LIMIT}`)
}

// Generate AI-powered recommendation using Gemini
async function generateAIRecommendation(symbol: string, marketData: any, newsData: any[]) {
  try {
    if (!checkQuotaAvailable()) {
      console.log(`[v0] Quota limit reached (${apiCallCount}/${DAILY_QUOTA_LIMIT}), using fallback for ${symbol}`)
      return generateFallbackRecommendation(symbol, marketData, newsData)
    }

    const GEMINI_API_KEY = "AIzaSyCnvqMspISmF_IAFkTO89czQkRxoKHLQzY"

    // Analyze news sentiment
    const newsTexts = newsData.map((article) => `${article.title} ${article.description || ""}`).join(" ")
    const sentiment = analyzeSentiment(newsTexts)

    // Create comprehensive context for AI
    const marketContext = marketData
      ? `
    Current Price: $${marketData.c || "N/A"}
    Volume: ${marketData.v || "N/A"}
    High: $${marketData.h || "N/A"}
    Low: $${marketData.l || "N/A"}
    `
      : "Market data unavailable"

    const newsContext =
      newsData.length > 0
        ? `
    Recent News Headlines:
    ${newsData
      .slice(0, 3)
      .map((article) => `- ${article.title}`)
      .join("\n")}
    
    News Sentiment: ${sentiment.label} (${sentiment.score.toFixed(2)})
    `
        : "No recent news available"

  const prompt = `As a professional retail analyst, analyze ${symbol} (product SKU) and provide a merchandising recommendation.

    MARKET DATA:
    ${marketContext}
    
    NEWS ANALYSIS:
    ${newsContext}
    
    CURRENT DATE: ${new Date().toLocaleDateString()}
    
    IMPORTANT: Respond ONLY with valid JSON in this exact format, no additional text or explanation:
    {
      "action": "PROMOTE",
      "confidence": 75,
      "reasoning": "Detailed analysis based on current data",
      "target_price": 150.00,
      "stop_loss": 140.00,
      "risk_level": "Medium",
      "time_horizon": "Medium-term"
    }
    
    Consider:
    - Current market conditions and volatility
    - News sentiment and recent developments
    - Technical indicators if available
    - Risk management principles
    
  Be conservative with recommendations during uncertain conditions. Action should be PROMOTE, DISCOUNT, or HOLD. (The API will map PROMOTE/DISCOUNT to BUY/SELL for compatibility where necessary.)`

    console.log(`[v0] Generating AI recommendation for ${symbol}...`)

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0.2, // Lower temperature for more consistent responses
            topK: 20,
            topP: 0.8,
            maxOutputTokens: 512, // Reduced token limit to save quota
          },
        }),
      },
    )

    if (!response.ok) {
      if (response.status === 429) {
        console.log(`[v0] Quota exceeded for ${symbol}, switching to fallback analysis`)
        // Mark quota as exhausted to prevent further API calls
        apiCallCount = DAILY_QUOTA_LIMIT
        return generateFallbackRecommendation(symbol, marketData, newsData)
      }

      // For other errors, log but don't throw - use fallback instead
      const errorText = await response.text()
      console.log(`[v0] Gemini API error for ${symbol} (${response.status}), using fallback analysis`)
      return generateFallbackRecommendation(symbol, marketData, newsData)
    }

    incrementQuotaUsage()

    const data = await response.json()
    const generatedText = data.candidates?.[0]?.content?.parts?.[0]?.text

    if (!generatedText) {
      console.log(`[v0] Empty AI response for ${symbol}, using fallback analysis`)
      return generateFallbackRecommendation(symbol, marketData, newsData)
    }

    console.log(`[v0] Raw AI response for ${symbol}:`, generatedText)

    try {
      // Clean the response text
      let cleanedText = generatedText.trim()
      cleanedText = cleanedText.replace(/```json\s*/g, "").replace(/```\s*/g, "")

      const jsonStart = cleanedText.indexOf("{")
      const jsonEnd = cleanedText.lastIndexOf("}")

      if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
        const jsonString = cleanedText.substring(jsonStart, jsonEnd + 1)
        const aiRecommendation = JSON.parse(jsonString)

        if (!aiRecommendation.action || !["BUY", "SELL", "HOLD"].includes(aiRecommendation.action.toUpperCase())) {
          console.log(`[v0] Invalid AI response format for ${symbol}, using fallback analysis`)
          return generateFallbackRecommendation(symbol, marketData, newsData)
        }

        return {
          symbol,
          action: aiRecommendation.action.toUpperCase(),
          confidence: Math.min(95, Math.max(60, aiRecommendation.confidence || 75)),
          reasoning: aiRecommendation.reasoning || "AI analysis based on current market data and news sentiment",
          target_price: aiRecommendation.target_price,
          stop_loss: aiRecommendation.stop_loss,
          risk_level: aiRecommendation.risk_level || "Medium",
          data_source: "AI_GENERATED",
          timestamp: new Date().toISOString(),
          market_data_available: !!marketData,
          news_sentiment: sentiment.label,
          news_count: newsData.length,
        }
      } else {
        console.log(`[v0] Could not parse AI response for ${symbol}, using fallback analysis`)
        return generateFallbackRecommendation(symbol, marketData, newsData)
      }
    } catch (parseError) {
        console.log(`[v0] JSON parse error for ${symbol}, using fallback analysis:`, getErrorMessage(parseError))
      return generateFallbackRecommendation(symbol, marketData, newsData)
    }
  } catch (error) {
    console.log(`[v0] AI service unavailable for ${symbol}, using fallback analysis:`, getErrorMessage(error))
    return generateFallbackRecommendation(symbol, marketData, newsData)
  }
}

export async function GET() {
  try {
    const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"
    console.log("[v0] Proxying recommendations request to backend...")

    const symbols = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "BNP"]
    const recommendations: any[] = []

    for (const symbol of symbols) {
      try {
        const res = await fetch(`${BACKEND_URL}/recommendations/${encodeURIComponent(symbol)}`)
        if (!res.ok) throw new Error(`Backend recommended ${res.status}`)
        const rec = await res.json()
        recommendations.push(rec)
      } catch (err) {
        console.error(`Error fetching recommendation for ${symbol}:`, err)
        recommendations.push({ symbol, action: "HOLD", confidence: 50, reasoning: "Unavailable", data_source: "BACKEND_ERROR" })
      }
    }

    return NextResponse.json({ recommendations, source: "proxy_backend", generated_at: new Date().toISOString() })
  } catch (error) {
    console.error("[v0] Critical error in recommendations:", error)

    return NextResponse.json(
      {
        recommendations: [],
        source: "error",
        generated_at: new Date().toISOString(),
        success_count: 0,
        fallback_count: 0,
        error_count: 6,
        status: "system_error",
        error: "Recommendation system temporarily unavailable",
        message: "Please try again later or check API quotas and keys",
      },
      { status: 503 },
    )
  }
}
