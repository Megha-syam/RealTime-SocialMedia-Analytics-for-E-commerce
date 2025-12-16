import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

const fallbackWords = [
  { text: "PRODUCTS", size: 48, color: "#999" },
  { text: "DEALS", size: 40, color: "#16a34a" },
  { text: "REVIEWS", size: 36, color: "#ef4444" },
  { text: "DISCOUNT", size: 34, color: "#f97316" },
  { text: "BESTSELLER", size: 32, color: "#0ea5e9" },
]

async function fetchLiveTrending(sku?: string) {
  try {
    const url = sku ? `${BACKEND_URL}/trending-live/${encodeURIComponent(sku)}` : `${BACKEND_URL}/trending-live`
    const res = await fetch(url)
    if (!res.ok) throw new Error(`Backend responded ${res.status}`)
    const data = await res.json()

    // Map returned keywords/counts into word cloud format
    const words = (data.keywords || []).map((k: string, i: number) => ({
      text: k.toUpperCase(),
      size: data.counts?.[k] || Math.max(10, 40 - i * 2),
      color: data.sentiment && data.sentiment.positive_pct > 50 ? "#16a34a" : "#ef4444",
    }))

    return words
  } catch (err) {
    console.error("Failed to fetch live trending from backend:", err)
    return null
  }
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url)
    const sku = url.searchParams.get("sku")

    const live = await fetchLiveTrending(sku || undefined)
    const words = live || fallbackWords

    return NextResponse.json({ words, timestamp: new Date().toISOString(), total_words: words.length })
  } catch (error) {
    console.error("Word cloud API error:", error)
    return NextResponse.json({ error: "Failed to generate word cloud data", words: [] }, { status: 500 })
  }
}
