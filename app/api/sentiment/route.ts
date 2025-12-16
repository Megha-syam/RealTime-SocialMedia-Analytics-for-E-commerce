import { NextResponse } from "next/server"

export async function GET() {
  try {
    // Simulate sentiment analysis with real-time calculation
    const currentTime = new Date()
    const baseScore = 0.5 + Math.sin((currentTime.getHours() / 24) * Math.PI * 2) * 0.3

    const sentimentData = {
      timestamp: currentTime.toISOString(),
      overall_sentiment: Math.max(0.1, Math.min(0.9, baseScore + (Math.random() - 0.5) * 0.2)),
      news_sentiment: Math.max(-1, Math.min(1, (Math.random() - 0.5) * 1.5)),
      social_sentiment: Math.max(-1, Math.min(1, (Math.random() - 0.5) * 1.2)),
      economic_sentiment: Math.max(-1, Math.min(1, (Math.random() - 0.5) * 0.8)),
      fear_greed_index: Math.round(Math.random() * 100),
    }

    return NextResponse.json(sentimentData)
  } catch (error) {
    console.error("Error fetching sentiment data:", error)
    return NextResponse.json({ error: "Failed to fetch sentiment data" }, { status: 500 })
  }
}
