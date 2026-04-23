# Sentiment Analysis Tab
### Route: `/sentiment`

---

## 1. Purpose

Sentiment Analysis tab **real-time financial news ko portfolio weights se connect karta hai**. Yeh tab live Google News headlines fetch karta hai, unhe FinBERT NLP model se analyze karta hai, aur phir sentiment scores ke basis par **stock weights adjust karta hai**.

**Project mein role:**
- Traditional RL models sirf **price data** dekhte hain — yeh tab **news aur language** ko bhi integrate karta hai
- Demonstrates ki **financial NLP (FinBERT)** real-time market signals provide kar sakta hai
- Portfolio allocation **sentiment-aware** ban jaata hai — bullish news = slight overweight, bearish = underweight
- Yeh tab project ka **most visually dynamic** component hai (live data, auto-refresh, animations)

---

## 2. Target Users & Usage

**Target Users:**
- Financial analysts tracking market sentiment
- Researchers studying NLP + portfolio integration
- Professors evaluating real-time data pipeline capability

**Real Usage Flow:**

```
User opens Sentiment tab
        ↓
System auto-fetches live news from Google News RSS (on mount)
        ↓
4 Metric Cards load: Headlines Count, Market Mood, Avg Score, Top Mover
        ↓
User reads News Feed (Tab 1) — real headlines with FinBERT scores
        ↓
Clicks any headline → expands to show full FinBERT breakdown
        ↓
Switches to Portfolio Impact tab (Tab 2)
        ↓
Sees which stocks got overweighted / underweighted due to sentiment
        ↓
Switches to Sectors tab (Tab 3)
        ↓
Sees which sectors are bullish vs bearish today
        ↓
User types custom text in manual analysis box → clicks Analyze
        ↓
Gets instant FinBERT score for their own text
        ↓
Auto-refresh runs every 3 minutes → "+N New" badge appears
```

---

## 3. Tools & Techniques

### 3.1 Frontend Stack
| Tool | Usage |
|------|-------|
| React 19 + TypeScript | Component UI |
| Recharts | LineChart (sentiment trend), BarChart (sector sentiment), PieChart (score distribution donut) |
| Framer Motion | News card expand/collapse, animated tab pill |
| localStorage | Session-persistent sentiment trend history (up to 48 data points) |
| Tailwind CSS v4 | Bullish=green, Bearish=red, Neutral=amber color system |

### 3.2 Backend APIs Called

| Endpoint | Method | Payload | Returns |
|---|---|---|---|
| `/api/news-sentiment` | GET | optional `force=true` | Live news + FinBERT scores + portfolio impact + sector breakdown |
| `/api/sentiment` | POST | `{text: string}` | Single text sentiment score |

**Backend files:** `src/sentiment/finbert.py`, `src/sentiment/news_fetcher.py`
**Timeout:** 120 seconds (FinBERT model loading + batch inference)

### 3.3 FinBERT Model

**What is FinBERT:**
- BERT model fine-tuned specifically on **financial text** (earnings calls, analyst reports, news)
- Model: `ProsusAI/finbert` (HuggingFace) — or local copy at `data/finbert_local/`
- Loaded once as singleton → cached in memory for fast subsequent calls

**Why FinBERT instead of general BERT/GPT:**
| Model | "Stock falls on earnings miss" | Accuracy |
|-------|-------------------------------|---------|
| General BERT | Neutral / Positive (misses context) | Low |
| FinBERT | Negative (understands financial language) | High |

**Output per headline:**
```json
{
  "score": -0.73,           // -1 (Bearish) to +1 (Bullish)
  "positive": 0.05,         // Probability
  "neutral": 0.17,          // Probability
  "negative": 0.78,         // Probability
  "label": "Bearish"
}
```

**Batch inference:** `predict_batch(texts, batch_size=16)` — processes all headlines efficiently

### 3.4 News Fetching — Google News RSS

**Pipeline:**
```
22 queries built:
  - 20 ticker-specific: "HDFCBANK NSE stock", "TCS NSE stock", ...
  - 2 market-wide: "NIFTY 50 index", "Indian stock market"
        ↓
Concurrent HTTP requests to Google News RSS
        ↓
feedparser parses RSS XML → headline, source, published_date
        ↓
Company name mapping: ticker → company name → query string
        ↓
Deduplicate by headline text
        ↓
Batch FinBERT inference on all headlines
        ↓
Results cached for 3 minutes (TTL)
```

**Example company mapping:**
```python
"HDFCBANK" → "HDFC Bank"
"TCS"      → "Tata Consultancy Services"
"RELIANCE" → "Reliance Industries"
```

### 3.5 Sentiment → Portfolio Weight Adjustment

**Algorithm:**
```
1. Aggregate FinBERT scores per stock (avg across all headlines for that ticker)
2. Normalize: sentiment_signal = score / max(|scores|)
3. Base weight = 1/N (equal weight baseline)
4. Adjusted weight = base_weight × (1 + α × sentiment_signal)
   where α = 0.1 (adjustment factor)
5. Re-normalize weights to sum to 1.0
```

**Effect:**
- Stock with sentiment = +0.8 → weight increases by ~8%
- Stock with sentiment = -0.6 → weight decreases by ~6%
- Stock with no news → weight stays at sector average

### 3.6 Auto-Refresh Mechanism

```javascript
// Runs on mount + every 3 minutes
useEffect(() => {
  const interval = setInterval(fetchNews, 3 * 60 * 1000);
  // Pauses when tab is not visible (document.hidden)
  return () => clearInterval(interval);
}, []);
```

- New headlines compared to previous fetch → "NEW" badge shown for 8 seconds
- "+N New" counter badge on refresh indicator
- "Xs ago" timestamp shows recency of last fetch

---

## 4. UI Components Breakdown

### 4.1 Metric Cards (4)
| Card | Value | Badges |
|------|-------|--------|
| Headlines Analyzed | Count (e.g., 47) | RICH SIGNAL (>30) / ADEQUATE / LOW SIGNAL |
| Market Mood | Bullish / Neutral / Bearish | BULLISH ↑ / NEUTRAL → / BEARISH ↓ |
| Avg Score | -1.0 to +1.0 decimal | STRONG +VE / POSITIVE / NEUTRAL / NEGATIVE / STRONG -VE |
| Top Mover | Ticker (e.g., TCS +3.2%) | OVERWEIGHT / UNDERWEIGHT |

### 4.2 Sentiment Trend Chart
- LineChart — session-long history of avg sentiment
- X-axis: time of each fetch
- Y-axis: sentiment score (-1 to +1)
- Reference line at y=0 (neutral boundary)
- Stored in localStorage — **persists across page refreshes**
- Up to 48 data points (4 hours of 3-min refresh)

### 4.3 Manual Analysis Card
- Textarea (500 char max) + Analyze button
- Enter key submits
- Result: badge (positive/negative/neutral) + score
- 3-column grid: Positive % / Neutral % / Negative %
- Score bar: visual -1 to +1 gradient bar with marker

### 4.4 Live Ticker Strip
- Horizontal scrolling strip of top 8 headlines
- Each: ticker pill (colored) · score · direction arrow · truncated headline
- Color: green strip (positive), red strip (negative), gray (neutral)

### 4.5 News Feed Tab (Tab 1)

**Collapsed card shows:**
- Score badge (color + up/neutral/down symbol + score%)
- First 2 lines of headline (truncated)
- Meta: ticker (colored border) · sector · published · "NEW" badge if recent

**Expanded card shows:**
- Full headline
- FinBERT verdict label
- Stacked probability bar (positive/neutral/negative colored segments)
- 3-column breakdown percentages
- Net score, confidence pill, source

### 4.6 Portfolio Impact Tab (Tab 2)
- Table: Stock · Sector · Sentiment Score · Base Weight % · Adjusted Weight % · Change %
- Change column: green/red color + up/down arrow
- Shows exactly how sentiment shifts allocations

### 4.7 Sectors Tab (Tab 3)
- Horizontal BarChart: sector sentiment scores
- Green bars (positive sectors), amber (neutral), red (negative)
- Sector detail cards below: sector name · n_headlines · avg_score · stacked bar

### 4.8 High Impact Alerts
- Auto-filtered: headlines with |score| > 0.3
- Shows top 8 strongest signals
- Left border + background color per sentiment
- Large score badge, full meta information

### 4.9 Score Distribution Donut (PieChart)
- 5 segments: Very Positive / Positive / Neutral / Negative / Very Negative
- Shows distribution of all analyzed headlines

---

## 5. Data Flow

```
Google News RSS (22 queries for NIFTY stocks)
        ↓
news_fetcher.py → feedparser → raw headlines
        ↓
Deduplication + ticker mapping
        ↓
FinBERT batch inference (ProsusAI/finbert)
        ↓
Per-headline: score, positive%, neutral%, negative%
        ↓
Aggregate per stock → sentiment_signal per ticker
        ↓
Weight adjustment: base_weight + α × sentiment_signal
        ↓
/api/news-sentiment response:
  news[] (headlines + scores)
  portfolio_impact[] (weight changes)
  sector_sentiment[] (per-sector avg)
  score_distribution (5-bucket breakdown)
        ↓
Sentiment.tsx → 3 tabs + charts + metric cards
        ↓
localStorage → trend history persisted
```

---

## 6. Edge Cases & Validations

| Scenario | Handling |
|----------|----------|
| No news found for a ticker | Excluded from portfolio impact; sector average used |
| FinBERT model not loaded | Returns error; backend shows "model loading" status |
| Google News RSS unreachable | Error state with retry; cached result shown if available |
| All headlines neutral | Market Mood = NEUTRAL, Avg Score ≈ 0, no weight changes |
| API timeout (120s) | Timeout error shown; "Try again later" message |
| Custom text > 512 tokens | FinBERT truncates to 512 tokens automatically |
| Tab hidden (background) | Auto-refresh paused (document.hidden API) |
| localStorage full | Old trend points cleared from front of array |

---

## 7. Performance Notes

- **FinBERT loading:** ~10–15 seconds first time → singleton cache → instant after
- **Batch inference (50 headlines):** ~3–5 seconds on CPU
- **3-minute TTL cache:** Prevents re-running FinBERT on same headlines
- **Concurrent news fetching:** All 22 RSS queries fire in parallel (asyncio)
- **120s timeout** on frontend — necessary for cold-start model loading

---

## 8. What Makes This Tab Impressive

- **Real live data** — actual Google News headlines, not mocked
- **FinBERT** — domain-specific financial NLP, not general-purpose sentiment
- **Auto-refresh every 3 minutes** — dashboard stays current without user interaction
- **3-way tabbed view** — News feed / Portfolio Impact / Sector breakdown — complete picture
- **localStorage persistence** — trend chart survives page reload
- **Manual analysis box** — professor can type any text and see live FinBERT output
- Connects NLP output directly to **portfolio allocation numbers** — not just "sentiment is positive"

---

*Tab: Sentiment Analysis | Route: `/sentiment` | File: `dashboard/src/pages/Sentiment.tsx`*
