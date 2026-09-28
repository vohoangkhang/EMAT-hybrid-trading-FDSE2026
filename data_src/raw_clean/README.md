# Raw Clean Data
Folder du lieu sach cho pipeline EMAT va cac mo hinh khac.

## Cau truc
```
raw_clean/
├── prices/          # Gia OHLCV dai han (yfinance)
│   ├── META_prices.csv
│   ├── GOOGL_prices.csv
│   ├── AAPL_prices.csv
│   ├── AMZN_prices.csv
│   ├── BABA_prices.csv
│   └── VNM_prices.csv
├── news_en/         # Tin tuc tieng Anh (Google News + Finnhub + Benzinga)
│   ├── all_en_news.csv       # Tat ca tin gop chung
│   ├── META_en_news.csv
│   ├── GOOGL_en_news.csv
│   ├── AAPL_en_news.csv
│   ├── AMZN_en_news.csv
│   ├── BABA_en_news.csv
│   └── VNM_en_news.csv
├── news_vi/         # Tin tuc tieng Viet (full-text)
│   ├── META_vi_news.csv
│   ├── GOOGL_vi_news.csv
│   └── VNM_vi_news.csv ...
├── sentiment_vi/    # Sentiment da danh gia san (tieng Viet)
│   ├── META_sentiment.csv
│   └── VNM_sentiment.csv ...
```

## Columns
- **prices**: Date, Open, High, Low, Close, Volume
- **news_en**: title, date, stock
- **news_vi**: Date, Content (+ cac cot khac tuy file)
- **sentiment_vi**: Date, Content, Sentiment, Confidence (+ cac cot khac)
