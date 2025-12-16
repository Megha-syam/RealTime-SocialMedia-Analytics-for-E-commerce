"use client"

import { useEffect, useRef } from "react"

interface WordData {
  text: string
  size: number
  color: string
}

interface WordCloudProps {
  words: WordData[]
  width?: number
  height?: number
}

export function WordCloud({ words, width = 400, height = 300 }: WordCloudProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || words.length === 0) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set canvas size
    canvas.width = width
    canvas.height = height

    // Sort words by size (largest first)
    const sortedWords = [...words].sort((a, b) => b.size - a.size)

    // Track placed words to avoid overlap
    const placedWords: Array<{ x: number; y: number; width: number; height: number }> = []

    sortedWords.forEach((word, index) => {
      const fontSize = Math.max(12, Math.min(48, word.size))
      ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
      ctx.fillStyle = word.color
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"

      const textMetrics = ctx.measureText(word.text)
      const textWidth = textMetrics.width
      const textHeight = fontSize

      let x, y
      let attempts = 0
      const maxAttempts = 50

      // Try to find a non-overlapping position
      do {
        x = Math.random() * (width - textWidth) + textWidth / 2
        y = Math.random() * (height - textHeight) + textHeight / 2
        attempts++
      } while (
        attempts < maxAttempts &&
        placedWords.some(
          (placed) =>
            x < placed.x + placed.width &&
            x + textWidth > placed.x &&
            y < placed.y + placed.height &&
            y + textHeight > placed.y,
        )
      )

      // Draw the word
      ctx.fillText(word.text, x, y)

      // Add to placed words
      placedWords.push({
        x: x - textWidth / 2,
        y: y - textHeight / 2,
        width: textWidth,
        height: textHeight,
      })
    })
  }, [words, width, height])

  return (
    <div className="flex items-center justify-center">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="border border-border rounded-lg bg-card"
        style={{ maxWidth: "100%", height: "auto" }}
      />
    </div>
  )
}
