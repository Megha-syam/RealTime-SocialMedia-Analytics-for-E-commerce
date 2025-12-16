-- Real-Time Social Media Analytics for E-Commerce Trends Database Schema
-- This script creates the necessary tables for storing social media mentions and sentiment analysis for e-commerce

-- CREATE DATABASE ecommerce_social_analytics;

-- \c ecommerce_social_analytics;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types
CREATE TYPE sentiment_label AS ENUM ('positive', 'negative', 'neutral');
CREATE TYPE recommendation_action AS ENUM ('BUY', 'SELL', 'HOLD');
CREATE TYPE risk_level AS ENUM ('Low', 'Medium', 'High');

-- Table: stocks
-- Stores information about tracked stocks/securities
CREATE TABLE IF NOT EXISTS stocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(255),
    sector VARCHAR(100),
    market_cap BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: news_articles
-- Stores news articles with sentiment analysis
CREATE TABLE IF NOT EXISTS news_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id VARCHAR(255) UNIQUE,
    title TEXT NOT NULL,
    description TEXT,
    content TEXT,
    source VARCHAR(100),
    author VARCHAR(255),
    url TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    sentiment_score DECIMAL(4,3), -- Range: -1.000 to 1.000
    sentiment_label sentiment_label,
    confidence DECIMAL(4,3), -- Range: 0.000 to 1.000
    vader_score DECIMAL(4,3),
    textblob_score DECIMAL(4,3),
    financial_score DECIMAL(4,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: economic_events
-- Stores economic calendar events
CREATE TABLE IF NOT EXISTS economic_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_name VARCHAR(255) NOT NULL,
    country VARCHAR(100),
    event_date TIMESTAMP WITH TIME ZONE,
    actual_value VARCHAR(50),
    forecast_value VARCHAR(50),
    previous_value VARCHAR(50),
    importance VARCHAR(20), -- Low, Medium, High
    currency VARCHAR(3),
    sentiment_score DECIMAL(4,3),
    sentiment_label sentiment_label,
    impact_level INTEGER, -- 1-5 scale
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: social_mentions
-- Stores social media mentions and sentiment
CREATE TABLE IF NOT EXISTS social_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES stocks(id) ON DELETE CASCADE,
    platform VARCHAR(50), -- twitter, reddit, etc.
    mention_id VARCHAR(255),
    content TEXT,
    author VARCHAR(255),
    followers_count INTEGER,
    likes_count INTEGER,
    retweets_count INTEGER,
    posted_at TIMESTAMP WITH TIME ZONE,
    sentiment_score DECIMAL(4,3),
    sentiment_label sentiment_label,
    confidence DECIMAL(4,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: sentiment_snapshots
-- Stores aggregated sentiment data for stocks at specific times
CREATE TABLE IF NOT EXISTS sentiment_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES stocks(id) ON DELETE CASCADE,
    snapshot_time TIMESTAMP WITH TIME ZONE NOT NULL,
    overall_sentiment DECIMAL(5,2), -- 0.00 to 100.00
    news_sentiment DECIMAL(5,2),
    social_sentiment DECIMAL(5,2),
    economic_sentiment DECIMAL(5,2),
    confidence DECIMAL(4,3),
    news_volume INTEGER DEFAULT 0,
    social_volume INTEGER DEFAULT 0,
    economic_events_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: ai_recommendations
-- Stores AI-generated trading recommendations
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES stocks(id) ON DELETE CASCADE,
    recommendation_action recommendation_action NOT NULL,
    confidence INTEGER CHECK (confidence >= 0 AND confidence <= 100),
    reasoning TEXT,
    risk_level risk_level,
    time_horizon VARCHAR(20), -- Short, Medium, Long
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: market_indicators
-- Stores key market indicators and economic data
CREATE TABLE IF NOT EXISTS market_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    indicator_name VARCHAR(100) NOT NULL,
    country VARCHAR(100),
    value DECIMAL(15,4),
    unit VARCHAR(20),
    date TIMESTAMP WITH TIME ZONE,
    previous_value DECIMAL(15,4),
    change_percent DECIMAL(6,3),
    sentiment_impact DECIMAL(4,3), -- Impact on market sentiment
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: data_refresh_log
-- Tracks data refresh operations
CREATE TABLE IF NOT EXISTS data_refresh_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_source VARCHAR(50) NOT NULL, -- news_api, trading_economics, social_media
    refresh_type VARCHAR(20) NOT NULL, -- scheduled, manual, error_recovery
    status VARCHAR(20) NOT NULL, -- success, failed, partial
    records_processed INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles(sentiment_score DESC);
CREATE INDEX IF NOT EXISTS idx_economic_events_date ON economic_events(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_social_mentions_stock_posted ON social_mentions(stock_id, posted_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_snapshots_stock_time ON sentiment_snapshots(stock_id, snapshot_time DESC);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_stock_active ON ai_recommendations(stock_id, is_active, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_indicators_date ON market_indicators(date DESC);
CREATE INDEX IF NOT EXISTS idx_data_refresh_log_source_started ON data_refresh_log(data_source, started_at DESC);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_stocks_updated_at BEFORE UPDATE ON stocks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_news_articles_updated_at BEFORE UPDATE ON news_articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_economic_events_updated_at BEFORE UPDATE ON economic_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
