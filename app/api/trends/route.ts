import { NextResponse } from "next/server"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const period = searchParams.get("period") || "1D"
    const sector = searchParams.get("sector") || "all"

    console.log(`[v0] Generating trends data for period: ${period}, sector: ${sector}`)

    const now = new Date()
    const dataPoints = []
    let timeUnit = "day"
    let dataCount = 7

    switch (period) {
      case "1H":
        timeUnit = "hour"
        dataCount = 24 // Last 24 hours
        break
      case "1D":
        timeUnit = "day"
        dataCount = 7 // Last 7 days
        break
      case "1M":
        timeUnit = "month"
        dataCount = 12 // Last 12 months
        break
      case "1Y":
        timeUnit = "year"
        dataCount = 5 // Last 5 years
        break
    }

    // Generate data based on time period
    for (let i = dataCount - 1; i >= 0; i--) {
      let date: Date
      let baseSentiment = 0.5
      let volume = 100

      switch (timeUnit) {
        case "hour":
          date = new Date(now.getTime() - i * 60 * 60 * 1000)
          // Hourly patterns - business hours have higher activity
          const hour = date.getHours()
          const isBusinessHour = hour >= 9 && hour <= 16
          baseSentiment = isBusinessHour ? 0.45 + Math.random() * 0.3 : 0.48 + Math.random() * 0.04
          volume = isBusinessHour ? 80 + Math.random() * 120 : 20 + Math.random() * 40
          break

        case "day":
          date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000)
          const dayOfWeek = date.getDay()
          const isWeekend = dayOfWeek === 0 || dayOfWeek === 6
          baseSentiment = isWeekend ? 0.48 + Math.random() * 0.04 : 0.45 + Math.random() * 0.3
          volume = isWeekend ? 20 + Math.random() * 50 : 80 + Math.random() * 120
          break

        case "month":
          date = new Date(now.getFullYear(), now.getMonth() - i, 1)
          // Monthly patterns - seasonal effects
          const month = date.getMonth()
          const isQ4 = month >= 9 // Q4 typically more volatile
          baseSentiment = isQ4 ? 0.4 + Math.random() * 0.4 : 0.45 + Math.random() * 0.3
          volume = isQ4 ? 150 + Math.random() * 200 : 100 + Math.random() * 150
          break

        case "year":
          date = new Date(now.getFullYear() - i, 0, 1)
          // Yearly patterns - economic cycles
          baseSentiment = 0.3 + Math.random() * 0.5
          volume = 200 + Math.random() * 300
          break
      }

      let sectorMultiplier = 1.0
      switch (sector) {
        case "technology":
          sectorMultiplier = 1.1 // Tech tends to be more optimistic
          baseSentiment *= sectorMultiplier
          volume *= 1.2 // Higher discussion volume
          break
        case "finance":
          sectorMultiplier = 0.95 // Finance more conservative
          baseSentiment *= sectorMultiplier
          volume *= 1.1
          break
        case "healthcare":
          sectorMultiplier = 1.05 // Healthcare steady growth
          baseSentiment *= sectorMultiplier
          volume *= 0.9
          break
        case "energy":
          sectorMultiplier = 0.9 // Energy more volatile/negative recently
          baseSentiment *= sectorMultiplier
          volume *= 1.3 // High volatility = high discussion
          break
        case "consumer":
          sectorMultiplier = 1.0 // Consumer neutral
          baseSentiment *= sectorMultiplier
          volume *= 1.0
          break
      }

      // Add trend factor
      const trendFactor = Math.sin((i / dataCount) * Math.PI) * 0.1
      const sentiment = Math.max(0.1, Math.min(0.9, baseSentiment + trendFactor))

      // Higher volume during extreme sentiment days
      if (sentiment > 0.7 || sentiment < 0.3) {
        volume += 50
      }

      dataPoints.push({
        timestamp: date.toISOString(),
        sentiment: Number(sentiment.toFixed(3)),
        volume: Math.floor(volume),
        period_type: timeUnit,
        sector: sector,
        volatility: Math.abs(sentiment - 0.5) * 2,
        trend_direction: sentiment > 0.6 ? "bullish" : sentiment < 0.4 ? "bearish" : "neutral",
      })
    }

    const avgSentiment = dataPoints.reduce((acc, curr) => acc + curr.sentiment, 0) / dataPoints.length
    const totalVolume = dataPoints.reduce((acc, curr) => acc + curr.volume, 0)
    const maxSentiment = Math.max(...dataPoints.map((d) => d.sentiment))
    const minSentiment = Math.min(...dataPoints.map((d) => d.sentiment))

    const response: any = {
      summary: {
        average_sentiment: Number(avgSentiment.toFixed(3)),
        total_volume: totalVolume,
        sentiment_range: {
          max: Number(maxSentiment.toFixed(3)),
          min: Number(minSentiment.toFixed(3)),
          spread: Number((maxSentiment - minSentiment).toFixed(3)),
        },
        market_status: `simulated_${timeUnit}_data`,
        data_quality: "simulated",
        last_updated: now.toISOString(),
        period: period,
        sector: sector,
      },
    }

    // Add data to appropriate time period key
    switch (period) {
      case "1H":
        response.hourly_data = dataPoints
        break
      case "1D":
        response.daily_data = dataPoints
        break
      case "1M":
        response.monthly_data = dataPoints
        break
      case "1Y":
        response.yearly_data = dataPoints
        break
      default:
        response.daily_data = dataPoints
    }

    console.log(`[v0] Generated ${dataPoints.length} ${timeUnit} trend data points for ${sector} sector`)
    console.log(`[v0] Average ${timeUnit} sentiment:`, avgSentiment.toFixed(3))
    console.log(`[v0] Total volume:`, totalVolume)

    return NextResponse.json(response)
  } catch (error) {
    console.error("[v0] Error generating trends:", error)
    return NextResponse.json(
      {
        error: "Failed to fetch trends",
        daily_data: [],
        hourly_data: [],
        monthly_data: [],
        yearly_data: [],
        summary: {
          average_sentiment: 0.5,
          total_volume: 0,
          sentiment_range: { max: 0.5, min: 0.5, spread: 0 },
          market_status: "error",
          data_quality: "error",
          last_updated: new Date().toISOString(),
        },
      },
      { status: 500 },
    )
  }
}
