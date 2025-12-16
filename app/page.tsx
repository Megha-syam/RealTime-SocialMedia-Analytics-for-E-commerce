"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"
import {
  Search,
  TrendingUp,
  Activity,
  Brain,
  RefreshCw,
  Calendar,
  Globe,
  MessageSquare,
  Cloud,
  Star,
  TrendingDown,
  Minus,
} from "lucide-react"
import { WordCloud } from "@/components/word-cloud"

interface SentimentData {
  timestamp: string
  overall_sentiment: number
  news_sentiment: number
  social_sentiment: number
  economic_sentiment: number
  fear_greed_index: number
}

interface NewsArticle {
  title: string
  source: string
  published_at: string
  sentiment_score: number
  sentiment_label: string
  url: string
}

interface Recommendation {
  symbol: string
  action: string
  confidence: number
  reasoning: string
  target_price?: number
  stop_loss?: number
  data_source?: string // Added data_source field to track AI vs fallback
  timestamp?: string // Added timestamp field
  error?: string // Added error field to handle errors
}

interface TrendData {
  time: string
  sentiment: number
  volume: number
}

export default function SentimentDashboard() {
  const [selectedProduct, setSelectedProduct] = useState("Iphone 17")
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null)
  const [newsData, setNewsData] = useState<NewsArticle[]>([])
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [trendData, setTrendData] = useState<TrendData[]>([])
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(new Date())
  const [loading, setLoading] = useState(true)
  const [wordCloudData, setWordCloudData] = useState<Array<{ text: string; size: number; color: string }>>([])
  const [timeFilter, setTimeFilter] = useState("1D")
  const [sectorFilter, setSectorFilter] = useState("all")
  const [quickSuggestion, setQuickSuggestion] = useState<string | null>(null)

  const API_BASE = "/api"

  const fetchSentimentData = async () => {
    try {
      const response = await fetch(`${API_BASE}/sentiment`)
      if (response.ok) {
        const data = await response.json()
        setSentimentData(data)
      }
    } catch (error) {
      console.error("Failed to fetch sentiment data:", error)
    }
  }

  const fetchNewsData = async () => {
    try {
      const response = await fetch(`${API_BASE}/news`)
      if (response.ok) {
        const data = await response.json()
        setNewsData(data.articles || [])
      }
    } catch (error) {
      console.error("Failed to fetch news data:", error)
    }
  }

  const fetchRecommendations = async () => {
    try {
      const response = await fetch(`${API_BASE}/recommendations`)
      if (response.ok) {
        const data = await response.json()
        setRecommendations(data.recommendations || [])
      }
    } catch (error) {
      console.error("Failed to fetch recommendations:", error)
    }
  }

  const getQuickSuggestion = (sentiment: number) => {
    if (sentiment >= 70) return "PROMOTE"
    if (sentiment <= 40) return "DISCOUNT"
    return "HOLD"
  }

  const getSentimentStars = (sentiment: number) => {
    return Math.round((sentiment / 100) * 5)
  }

  const renderStarRating = (rating: number) => {
    return (
      <div className="flex items-center gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`h-4 w-4 ${star <= rating ? "fill-yellow-400 text-yellow-400" : "text-gray-300"}`}
          />
        ))}
        <span className="ml-1 text-sm text-muted-foreground">({rating}/5)</span>
      </div>
    )
  }

  const fetchTrendData = async () => {
    try {
      const response = await fetch(`${API_BASE}/trends?period=${timeFilter}&sector=${sectorFilter}`)
      if (response.ok) {
        const data = await response.json()
        let transformedData = []

        if (timeFilter === "1H") {
          transformedData =
            data.hourly_data?.map((item: any) => ({
              time: new Date(item.timestamp).toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
              }),
              sentiment: Math.round(item.sentiment * 100),
              volume: item.volume || 0,
            })) || []
        } else if (timeFilter === "1M") {
          transformedData =
            data.monthly_data?.map((item: any) => ({
              time: new Date(item.timestamp).toLocaleDateString("en-US", {
                month: "short",
                year: "2-digit",
              }),
              sentiment: Math.round(item.sentiment * 100),
              volume: item.volume || 0,
            })) || []
        } else if (timeFilter === "1Y") {
          transformedData =
            data.yearly_data?.map((item: any) => ({
              time: new Date(item.timestamp).toLocaleDateString("en-US", {
                month: "short",
                year: "numeric",
              }),
              sentiment: Math.round(item.sentiment * 100),
              volume: item.volume || 0,
            })) || []
        } else {
          transformedData =
            data.daily_data?.map((item: any) => ({
              time: new Date(item.timestamp).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              }),
              sentiment: Math.round(item.sentiment * 100),
              volume: item.volume || 0,
            })) || []
        }

        setTrendData(transformedData)
      }
    } catch (error) {
      console.error("Failed to fetch trend data:", error)
    }
  }

  const fetchWordCloudData = async () => {
    try {
      const response = await fetch(`${API_BASE}/wordcloud`)
      if (response.ok) {
        const data = await response.json()
        setWordCloudData(data.words || [])
      }
    } catch (error) {
      console.error("Failed to fetch word cloud data:", error)
    }
  }

  const fetchAllData = async () => {
    setIsRefreshing(true)
    setLoading(true)

    await Promise.all([
      fetchSentimentData(),
      fetchNewsData(),
      fetchRecommendations(),
      fetchTrendData(),
      fetchWordCloudData(),
    ])

    if (sentimentData) {
      setQuickSuggestion(getQuickSuggestion(sentimentData.overall_sentiment * 100))
    }

    setLastUpdate(new Date())
    setIsRefreshing(false)
    setLoading(false)
  }

  useEffect(() => {
    fetchAllData()
  }, [timeFilter, sectorFilter])

  const handleRefresh = () => {
    fetchAllData()
  }

  const getSentimentColor = (sentiment: string | number) => {
    if (typeof sentiment === "string") {
      switch (sentiment.toLowerCase()) {
        case "positive":
        case "bullish":
          return "text-success"
        case "negative":
        case "bearish":
          return "text-destructive"
        default:
          return "text-muted-foreground"
      }
    }
    if (sentiment >= 70) return "text-success"
    if (sentiment <= 40) return "text-destructive"
    return "text-warning"
  }

  const getSentimentBadge = (sentiment: string) => {
    const variants = {
      promote: "bg-success/10 text-success border-success/20",
      discount: "bg-destructive/10 text-destructive border-destructive/20",
      hold: "bg-warning/10 text-warning border-warning/20",
      positive: "bg-success/10 text-success border-success/20",
      bullish: "bg-success/10 text-success border-success/20",
      negative: "bg-destructive/10 text-destructive border-destructive/20",
      bearish: "bg-destructive/10 text-destructive border-destructive/20",
      neutral: "bg-muted text-muted-foreground border-border",
    }
    return variants[sentiment.toLowerCase() as keyof typeof variants] || variants.neutral
  }

  if (loading && !sentimentData) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading real-time e-commerce data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                <Brain className="h-8 w-8 text-primary" />
                <h1 className="text-2xl font-bold text-foreground">Real-Time Social Media Analytics</h1>
              </div>
              <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
                Live
              </Badge>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm text-muted-foreground">Last updated: {lastUpdate.toLocaleTimeString()}</div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleRefresh}
                disabled={isRefreshing}
                className="gap-2 bg-transparent"
              >
                <RefreshCw className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-6">
        {/* Search Bar with Quick Suggestion */}
        <div className="mb-6">
          <div className="flex items-center gap-4 max-w-2xl">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                placeholder="Enter product SKU or keyword (e.g., SKU-XYZ-01, iphone13, nike shoes)"
                value={selectedProduct}
                onChange={(e) => setSelectedProduct(e.target.value.toUpperCase())}
                className="pl-10"
              />
            </div>
            {quickSuggestion && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Quick Suggestion:</span>
                <Badge
                  className={
                    quickSuggestion === "PROMOTE"
                      ? "bg-success/10 text-success border-success/20"
                      : quickSuggestion === "DISCOUNT"
                        ? "bg-destructive/10 text-destructive border-destructive/20"
                        : "bg-warning/10 text-warning border-warning/20"
                  }
                >
                  {quickSuggestion === "PROMOTE" && <TrendingUp className="h-3 w-3 mr-1" />}
                  {quickSuggestion === "DISCOUNT" && <TrendingDown className="h-3 w-3 mr-1" />}
                  {quickSuggestion === "HOLD" && <Minus className="h-3 w-3 mr-1" />}
                  {quickSuggestion}
                </Badge>
              </div>
            )}
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Overall Sentiment</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-balance">
                <span className={getSentimentColor(sentimentData?.overall_sentiment || 0)}>
                  {Math.round((sentimentData?.overall_sentiment || 0) * 100)}/100
                </span>
              </div>
              <Progress value={(sentimentData?.overall_sentiment || 0) * 100} className="mt-2" />
              <div className="mt-2">
                {renderStarRating(getSentimentStars((sentimentData?.overall_sentiment || 0) * 100))}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {(sentimentData?.overall_sentiment || 0) >= 0.7
                  ? "Strong Demand"
                  : (sentimentData?.overall_sentiment || 0) <= 0.4
                    ? "Weak Demand"
                    : "Stable Demand"} {" "}
                product sentiment
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">News Sentiment</CardTitle>
              <Globe className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${getSentimentColor(sentimentData?.news_sentiment || 0)}`}>
                {sentimentData?.news_sentiment
                  ? (sentimentData.news_sentiment > 0 ? "+" : "") + sentimentData.news_sentiment.toFixed(2)
                  : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">
                <TrendingUp className="inline h-3 w-3 mr-1" />
                From latest retail & social analysis
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Social Sentiment</CardTitle>
              <MessageSquare className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${getSentimentColor(sentimentData?.social_sentiment || 0)}`}>
                {sentimentData?.social_sentiment
                  ? (sentimentData.social_sentiment > 0 ? "+" : "") + sentimentData.social_sentiment.toFixed(2)
                  : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">
                <MessageSquare className="inline h-3 w-3 mr-1" />
                Social media analysis
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Fear & Greed Index</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${getSentimentColor(sentimentData?.fear_greed_index || 0)}`}>
                {sentimentData?.fear_greed_index ? Math.round(sentimentData.fear_greed_index) : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">
                <Activity className="inline h-3 w-3 mr-1" />
                Retail demand indicator
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="news">News Feed</TabsTrigger>
            <TabsTrigger value="recommendations">AI Recommendations</TabsTrigger>
            <TabsTrigger value="trends">Trend Analysis</TabsTrigger>
            <TabsTrigger value="wordcloud">Word Cloud</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="flex items-center gap-4 mb-4">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Time Period:</span>
                <Select value={timeFilter} onValueChange={setTimeFilter}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1H">Hourly</SelectItem>
                    <SelectItem value="1D">Daily</SelectItem>
                    <SelectItem value="1M">Monthly</SelectItem>
                    <SelectItem value="1Y">Yearly</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Sector:</span>
                <Select value={sectorFilter} onValueChange={setSectorFilter}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Sectors</SelectItem>
                    <SelectItem value="technology">Technology</SelectItem>
                    <SelectItem value="finance">Finance</SelectItem>
                    <SelectItem value="healthcare">Healthcare</SelectItem>
                    <SelectItem value="energy">Energy</SelectItem>
                    <SelectItem value="consumer">Consumer</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Sentiment Trend Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Sentiment Trend</CardTitle>
                  <CardDescription>
                    {timeFilter === "1H"
                      ? "Hourly"
                      : timeFilter === "1M"
                        ? "Monthly"
                        : timeFilter === "1Y"
                          ? "Yearly"
                          : "Daily"}{" "}
                    sentiment analysis for {selectedProduct}
                    {sectorFilter !== "all" && ` (${sectorFilter} sector)`}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={trendData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                      <XAxis dataKey="time" className="text-muted-foreground" />
                      <YAxis className="text-muted-foreground" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="sentiment"
                        stroke="red"
                        strokeWidth={3}
                        dot={{
                          fill: "red",
                          stroke: "black",
                          strokeWidth: 2,
                          r: 6,
                        }}
                        activeDot={{
                          r: 8,
                          fill: "red",
                          stroke: "black",
                          strokeWidth: 3,
                        }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Volume Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle>Discussion Volume</CardTitle>
                  <CardDescription>
                    {timeFilter === "1H"
                      ? "Hourly"
                      : timeFilter === "1M"
                        ? "Monthly"
                        : timeFilter === "1Y"
                          ? "Yearly"
                          : "Daily"}{" "}
                    news and social media mention volume
                    {sectorFilter !== "all" && ` (${sectorFilter} sector)`}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={trendData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                      <XAxis dataKey="time" className="text-muted-foreground" />
                      <YAxis className="text-muted-foreground" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Bar dataKey="volume" fill="red" stroke="black" strokeWidth={1} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Word Cloud Preview */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cloud className="h-5 w-5" />
                  Trending Keywords
                </CardTitle>
                <CardDescription>Most frequently mentioned terms in retail and e-commerce discussions</CardDescription>
              </CardHeader>
              <CardContent>
                {wordCloudData.length > 0 ? (
                  <WordCloud words={wordCloudData} width={800} height={200} />
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Cloud className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>Loading trending keywords...</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="news" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Live News Feed</CardTitle>
                <CardDescription>Latest retail and social posts with sentiment analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {newsData.length > 0 ? (
                  newsData.map((article, index) => (
                    <div key={index} className="flex items-start justify-between p-4 border border-border rounded-lg">
                      <div className="flex-1">
                        <h3 className="font-medium text-balance leading-relaxed">{article.title}</h3>
                        <div className="flex items-center gap-3 mt-2 text-sm text-muted-foreground">
                          <span>{article.source}</span>
                          <span>•</span>
                          <span>{new Date(article.published_at).toLocaleString()}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 ml-4">
                        <Badge className={getSentimentBadge(article.sentiment_label)}>{article.sentiment_label}</Badge>
                        <div className={`text-sm font-medium ${getSentimentColor(article.sentiment_score)}`}>
                          {article.sentiment_score > 0 ? "+" : ""}
                          {article.sentiment_score.toFixed(1)}
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No news data available. Please check your API connection.
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="recommendations" className="space-y-4">
            <div className="grid gap-4">
              {recommendations.length > 0 ? (
                recommendations.map((rec, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">{rec.symbol}</CardTitle>
                        <div className="flex items-center gap-2">
                          {rec.action ? (
                            <Badge
                              className={
                                rec.action === "PROMOTE" || rec.action === "BUY"
                                  ? "bg-success/10 text-success border-success/20"
                                  : rec.action === "DISCOUNT" || rec.action === "SELL"
                                    ? "bg-destructive/10 text-destructive border-destructive/20"
                                    : "bg-warning/10 text-warning border-warning/20"
                              }
                            >
                              {rec.action}
                            </Badge>
                          ) : (
                            <Badge className="bg-destructive/10 text-destructive border-destructive/20">
                              UNAVAILABLE
                            </Badge>
                          )}
                          {rec.confidence && (
                            <div className="text-sm text-muted-foreground">{rec.confidence}% confidence</div>
                          )}
                          {rec.data_source === "AI_GENERATED" ? (
                            <Badge variant="outline" className="bg-success/10 text-success border-success/20 text-xs">
                              AI Generated ✓
                            </Badge>
                          ) : rec.data_source === "FALLBACK_ANALYSIS" ? (
                            <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20 text-xs">
                              Fallback Analysis
                            </Badge>
                          ) : rec.data_source === "API_ERROR" ? (
                            <Badge
                              variant="outline"
                              className="bg-destructive/10 text-destructive border-destructive/20 text-xs"
                            >
                              API Error
                            </Badge>
                          ) : rec.data_source === "PROCESSING_ERROR" ? (
                            <Badge
                              variant="outline"
                              className="bg-destructive/10 text-destructive border-destructive/20 text-xs"
                            >
                              Processing Error
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="bg-warning/10 text-warning border-warning/20 text-xs">
                              Service Unavailable
                            </Badge>
                          )}
                        </div>
                      </div>
                      {rec.timestamp && (
                        <CardDescription className="text-xs">
                          Generated: {new Date(rec.timestamp).toLocaleString()}
                        </CardDescription>
                      )}
                    </CardHeader>
                    <CardContent>
                      {rec.confidence && <Progress value={rec.confidence} className="mb-3" />}
                      {rec.reasoning ? (
                        <p className="text-sm text-muted-foreground leading-relaxed">{rec.reasoning}</p>
                      ) : rec.error ? (
                        <div className="text-sm text-destructive leading-relaxed">
                          <strong>Error:</strong> {rec.error}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          No analysis available for {rec.symbol}
                        </p>
                      )}
                      {rec.target_price && (
                        <div className="mt-2 text-sm">
                          <span className="text-muted-foreground">Target: </span>
                          <span className="font-medium">${rec.target_price}</span>
                          {rec.stop_loss && (
                            <>
                              <span className="text-muted-foreground ml-4">Stop Loss: </span>
                              <span className="font-medium">${rec.stop_loss}</span>
                            </>
                          )}
                        </div>
                      )}
                      {rec.data_source === "FALLBACK_ANALYSIS" && (
                        <div className="mt-3 p-3 bg-primary/10 border border-primary/20 rounded text-sm text-primary">
                          <strong>ℹ️ Fallback Analysis</strong>
                          <br />
                          This recommendation was generated using rule-based sentiment analysis due to AI service
                          limitations. While reliable, it may be less sophisticated than full AI analysis.
                        </div>
                      )}
                      {rec.data_source === "API_ERROR" && rec.error?.includes("QUOTA_EXCEEDED") && (
                        <div className="mt-3 p-3 bg-destructive/10 border border-destructive/20 rounded text-sm text-destructive">
                          <strong>⚠️ API Quota Exceeded</strong>
                          <br />
                          The Gemini AI service has reached its daily quota limit. The system has automatically switched
                          to fallback analysis for reliable recommendations.
                        </div>
                      )}
                      {rec.data_source === "API_ERROR" && !rec.error?.includes("QUOTA_EXCEEDED") && (
                        <div className="mt-3 p-3 bg-destructive/10 border border-destructive/20 rounded text-sm text-destructive">
                          <strong>⚠️ AI Service Error</strong>
                          <br />
                          The AI recommendation service is temporarily unavailable. Please check your API configuration
                          or try again later.
                        </div>
                      )}
                      {rec.data_source === "PROCESSING_ERROR" && (
                        <div className="mt-3 p-3 bg-warning/10 border border-warning/20 rounded text-sm text-warning">
                          <strong>⚠️ Processing Error</strong>
                          <br />
                          Unable to process data for {rec.symbol}. This may be due to network issues or data source
                          limitations.
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))
              ) : (
                <Card>
                  <CardContent className="text-center py-8">
                    <div className="text-muted-foreground mb-4">
                      <Brain className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p className="text-lg font-medium">No AI Recommendations Available</p>
                      <p className="text-sm">
                        The recommendation service is currently unavailable. Please check your API configuration and try
                        refreshing.
                      </p>
                    </div>
                    <Button onClick={handleRefresh} variant="outline" className="gap-2 bg-transparent">
                      <RefreshCw className="h-4 w-4" />
                      Retry
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="trends" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Retail Analysis</CardTitle>
                <CardDescription>Real-time retail sentiment trends and patterns</CardDescription>
              </CardHeader>
              <CardContent>
                {trendData.length > 0 ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-success">
                          {trendData[trendData.length - 1]?.sentiment || 0}
                        </div>
                        <div className="text-sm text-muted-foreground">Current Sentiment</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">
                          {Math.round(trendData.reduce((acc, curr) => acc + curr.sentiment, 0) / trendData.length)}
                        </div>
                        <div className="text-sm text-muted-foreground">Average Sentiment</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-warning">
                          {Math.max(...trendData.map((d) => d.volume))}
                        </div>
                        <div className="text-sm text-muted-foreground">Peak Volume</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-muted-foreground">{trendData.length}</div>
                        <div className="text-sm text-muted-foreground">Data Points</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">Loading trend analysis...</div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="wordcloud" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cloud className="h-6 w-6" />
                  Retail Keywords Word Cloud
                </CardTitle>
                <CardDescription>
                  Visual representation of the most frequently mentioned terms in retail and social media
                  discussions
                </CardDescription>
              </CardHeader>
              <CardContent>
                {wordCloudData.length > 0 ? (
                  <div className="space-y-6">
                    <WordCloud words={wordCloudData} width={1000} height={400} />
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">{wordCloudData.length}</div>
                        <div className="text-muted-foreground">Total Keywords</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-success">
                          {wordCloudData.filter((w) => w.color.includes("success")).length}
                        </div>
                        <div className="text-muted-foreground">Positive Terms</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-destructive">
                          {wordCloudData.filter((w) => w.color.includes("destructive")).length}
                        </div>
                        <div className="text-muted-foreground">Negative Terms</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-warning">
                          {wordCloudData.filter((w) => w.color.includes("warning")).length}
                        </div>
                        <div className="text-muted-foreground">Neutral Terms</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Cloud className="h-16 w-16 mx-auto mb-4 opacity-50 text-muted-foreground" />
                    <p className="text-lg font-medium text-muted-foreground mb-2">No Word Cloud Data Available</p>
                    <p className="text-sm text-muted-foreground mb-4">
                      Word cloud data is currently being processed or unavailable.
                    </p>
                    <Button onClick={handleRefresh} variant="outline" className="gap-2 bg-transparent">
                      <RefreshCw className="h-4 w-4" />
                      Refresh Data
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
