# Stock Market Sentiment & Price Direction Analysis

A beginner-friendly portfolio project combining daily stock-price data, news sentiment,
technical indicators and a machine-learning baseline to classify next-day price direction.

> **Important:** The included CSV files are synthetic demonstration data. They are not
> real market prices or real news, and model results from them must not be interpreted
> as evidence of a profitable trading strategy. This project is for learning and
> portfolio demonstration only, not investment advice.

## What this project demonstrates
- Loading and joining price and news datasets with pandas
- News sentiment scoring with VADER
- Feature engineering: daily returns, moving averages, RSI and MACD
- Chronological train/test split (no random shuffling)
- Logistic Regression baseline and classification metrics
- Visualising price, sentiment and model results

## Folder structure
```text
stock-market-sentiment-price-prediction/
├── data/
│   ├── sample_stock_prices.csv
│   └── sample_news_headlines.csv
├── src/
│   └── train_model.py
├── README.md
└── requirements.txt
```

## Setup
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\\Scripts\\activate

pip install -r requirements.txt
python src/train_model.py
```

## Replace demo data with real data
1. Obtain daily OHLCV data from a permitted provider (for example, Yahoo Finance via
   `yfinance`, subject to its terms and availability).
2. Prepare a CSV with columns: `date, open, high, low, close, volume`.
3. Obtain news headlines from a source whose licence/terms permit your use.
4. Prepare a CSV with columns: `date, headline`.
5. Replace the two sample CSV files, retaining the column names.

For a fair evaluation, use headlines published before or during the relevant trading
session and align them carefully with the market calendar. Avoid using future news
or future prices in features.

## Model and evaluation
The baseline predicts whether the **next trading day's close** is higher than the
current day's close. It uses a chronological holdout split. The script prints accuracy,
precision, recall, F1 and a confusion matrix. These metrics can vary with the data.

## Suggested improvements
- Compare Logistic Regression with Random Forest and XGBoost.
- Add FinBERT sentiment and compare it with VADER.
- Add walk-forward validation and compare against a majority-class baseline.
- Add transaction costs and realistic backtesting only after validating the data pipeline.
- Build a Power BI dashboard for price trends, sentiment and model evaluation.

## Interview explanation
"I built an end-to-end learning project that combines daily market data with news
sentiment. I cleaned and merged the datasets, engineered technical indicators,
scored headlines using VADER, and trained a Logistic Regression baseline to classify
next-day price direction. I used a chronological split to reduce look-ahead leakage.
The bundled data is synthetic, so I present it as a technical demonstration rather
than a validated trading model."
