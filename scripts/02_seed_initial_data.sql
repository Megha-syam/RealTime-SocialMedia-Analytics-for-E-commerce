-- Seed initial data for Real-Time Social Media Analytics for E-Commerce Trends

-- Insert popular stocks to track
INSERT INTO stocks (symbol, name, sector, market_cap) VALUES
('AAPL', 'Apple Inc.', 'Technology', 3000000000000),
('MSFT', 'Microsoft Corporation', 'Technology', 2800000000000),
('GOOGL', 'Alphabet Inc.', 'Technology', 1700000000000),
('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 1500000000000),
('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 800000000000),
('NVDA', 'NVIDIA Corporation', 'Technology', 1800000000000),
('META', 'Meta Platforms Inc.', 'Technology', 800000000000),
('NFLX', 'Netflix Inc.', 'Communication Services', 200000000000),
('AMD', 'Advanced Micro Devices', 'Technology', 240000000000),
('CRM', 'Salesforce Inc.', 'Technology', 250000000000),
('ORCL', 'Oracle Corporation', 'Technology', 300000000000),
('ADBE', 'Adobe Inc.', 'Technology', 220000000000),
('PYPL', 'PayPal Holdings Inc.', 'Financial Services', 80000000000),
('INTC', 'Intel Corporation', 'Technology', 200000000000),
('CSCO', 'Cisco Systems Inc.', 'Technology', 200000000000)
ON CONFLICT (symbol) DO NOTHING;

-- Insert sample market indicators
INSERT INTO market_indicators (indicator_name, country, value, unit, date, sentiment_impact) VALUES
('GDP Growth Rate', 'United States', 2.1, '%', CURRENT_TIMESTAMP - INTERVAL '1 day', 0.3),
('Unemployment Rate', 'United States', 3.7, '%', CURRENT_TIMESTAMP - INTERVAL '1 day', -0.1),
('Inflation Rate', 'United States', 3.2, '%', CURRENT_TIMESTAMP - INTERVAL '1 day', -0.4),
('Federal Funds Rate', 'United States', 5.25, '%', CURRENT_TIMESTAMP - INTERVAL '1 day', -0.2),
('Consumer Confidence Index', 'United States', 102.0, 'Index', CURRENT_TIMESTAMP - INTERVAL '1 day', 0.2),
('VIX Volatility Index', 'United States', 18.5, 'Index', CURRENT_TIMESTAMP - INTERVAL '1 day', -0.3)
ON CONFLICT DO NOTHING;

-- Insert sample economic events
INSERT INTO economic_events (event_name, country, event_date, importance, sentiment_score, sentiment_label) VALUES
('Federal Reserve Interest Rate Decision', 'United States', CURRENT_TIMESTAMP + INTERVAL '7 days', 'High', 0.1, 'neutral'),
('Non-Farm Payrolls', 'United States', CURRENT_TIMESTAMP + INTERVAL '3 days', 'High', 0.0, 'neutral'),
('Consumer Price Index', 'United States', CURRENT_TIMESTAMP + INTERVAL '10 days', 'High', 0.0, 'neutral'),
('Gross Domestic Product', 'United States', CURRENT_TIMESTAMP + INTERVAL '14 days', 'High', 0.0, 'neutral'),
('Retail Sales', 'United States', CURRENT_TIMESTAMP + INTERVAL '5 days', 'Medium', 0.0, 'neutral')
ON CONFLICT DO NOTHING;

-- Create a view for latest sentiment data
CREATE OR REPLACE VIEW latest_sentiment_view AS
SELECT 
    s.symbol,
    s.name,
    ss.overall_sentiment,
    ss.news_sentiment,
    ss.social_sentiment,
    ss.economic_sentiment,
    ss.confidence,
    ss.news_volume,
    ss.social_volume,
    ss.snapshot_time,
    ar.recommendation_action,
    ar.confidence as recommendation_confidence,
    ar.reasoning as recommendation_reasoning,
    ar.generated_at as recommendation_date
FROM stocks s
LEFT JOIN LATERAL (
    SELECT * FROM sentiment_snapshots 
    WHERE stock_id = s.id 
    ORDER BY snapshot_time DESC 
    LIMIT 1
) ss ON true
LEFT JOIN LATERAL (
    SELECT * FROM ai_recommendations 
    WHERE stock_id = s.id AND is_active = true
    ORDER BY generated_at DESC 
    LIMIT 1
) ar ON true;

-- Create a view for trending news
CREATE OR REPLACE VIEW trending_news_view AS
SELECT 
    na.title,
    na.description,
    na.source,
    na.url,
    na.published_at,
    na.sentiment_score,
    na.sentiment_label,
    na.confidence,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - na.published_at))/3600 as hours_ago
FROM news_articles na
WHERE na.published_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY na.published_at DESC, ABS(na.sentiment_score) DESC
LIMIT 50;

-- Create a function to get sentiment trend for a stock
CREATE OR REPLACE FUNCTION get_sentiment_trend(stock_symbol VARCHAR, hours_back INTEGER DEFAULT 24)
RETURNS TABLE (
    snapshot_time TIMESTAMP WITH TIME ZONE,
    overall_sentiment DECIMAL(5,2),
    news_sentiment DECIMAL(5,2),
    social_sentiment DECIMAL(5,2),
    volume INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ss.snapshot_time,
        ss.overall_sentiment,
        ss.news_sentiment,
        ss.social_sentiment,
        (ss.news_volume + ss.social_volume) as volume
    FROM sentiment_snapshots ss
    JOIN stocks s ON ss.stock_id = s.id
    WHERE s.symbol = stock_symbol
    AND ss.snapshot_time >= CURRENT_TIMESTAMP - (hours_back || ' hours')::INTERVAL
    ORDER BY ss.snapshot_time ASC;
END;
$$ LANGUAGE plpgsql;

-- Create a function to calculate market sentiment summary
CREATE OR REPLACE FUNCTION get_market_sentiment_summary()
RETURNS TABLE (
    total_stocks INTEGER,
    bullish_count INTEGER,
    bearish_count INTEGER,
    neutral_count INTEGER,
    avg_sentiment DECIMAL(5,2),
    market_mood VARCHAR(20)
) AS $$
DECLARE
    bullish_threshold DECIMAL := 60.0;
    bearish_threshold DECIMAL := 40.0;
BEGIN
    RETURN QUERY
    WITH latest_sentiments AS (
        SELECT DISTINCT ON (s.id) 
            s.id,
            ss.overall_sentiment
        FROM stocks s
        JOIN sentiment_snapshots ss ON s.id = ss.stock_id
        ORDER BY s.id, ss.snapshot_time DESC
    )
    SELECT 
        COUNT(*)::INTEGER as total_stocks,
        COUNT(CASE WHEN overall_sentiment >= bullish_threshold THEN 1 END)::INTEGER as bullish_count,
        COUNT(CASE WHEN overall_sentiment <= bearish_threshold THEN 1 END)::INTEGER as bearish_count,
        COUNT(CASE WHEN overall_sentiment > bearish_threshold AND overall_sentiment < bullish_threshold THEN 1 END)::INTEGER as neutral_count,
        ROUND(AVG(overall_sentiment), 2) as avg_sentiment,
        CASE 
            WHEN AVG(overall_sentiment) >= bullish_threshold THEN 'Bullish'
            WHEN AVG(overall_sentiment) <= bearish_threshold THEN 'Bearish'
            ELSE 'Neutral'
        END as market_mood
    FROM latest_sentiments;
END;
$$ LANGUAGE plpgsql;
