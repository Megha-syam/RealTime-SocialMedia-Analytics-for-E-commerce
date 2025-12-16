import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/news?limit=20`)
    if (!res.ok) throw new Error(`Backend news responded ${res.status}`)
    const data = await res.json()

    // Expect backend to return { news: [...] }
    const articles = data.news || []
    return NextResponse.json({ articles })
  } catch (err) {
    console.error("Error fetching proxied news:", err)
    const fallbackArticles = [
      {
        title: "No live news available",
        source: "Local",
        published_at: new Date().toISOString(),
        sentiment_score: 0,
        sentiment_label: "neutral",
        url: "#",
      },
    ]
    return NextResponse.json({ articles: fallbackArticles })
  }
}
