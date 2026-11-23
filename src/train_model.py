from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def main():
    prices = pd.read_csv(DATA / "sample_stock_prices.csv", parse_dates=["date"])
    news = pd.read_csv(DATA / "sample_news_headlines.csv", parse_dates=["date"])

    analyzer = SentimentIntensityAnalyzer()
    news["sentiment"] = news["headline"].fillna("").apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    daily_sentiment = news.groupby("date", as_index=False)["sentiment"].mean()

    df = prices.merge(daily_sentiment, on="date", how="left").sort_values("date")
    df["sentiment"] = df["sentiment"].fillna(0)
    df["daily_return"] = df["close"].pct_change()
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["rsi_14"] = calculate_rsi(df["close"])
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # Target: next trading day's direction. Features are known on current day.
    df["target_next_day_up"] = (df["close"].shift(-1) > df["close"]).astype(float)\n    df.loc[df.index[-1], "target_next_day_up"] = float("nan")
    feature_cols = ["sentiment", "daily_return", "sma_5", "sma_10", "rsi_14", "macd", "volume"]
    model_data = df.dropna(subset=feature_cols + ["target_next_day_up"]).copy()

    split = int(len(model_data) * 0.8)
    train, test = model_data.iloc[:split], model_data.iloc[split:]
    X_train, y_train = train[feature_cols], train["target_next_day_up"]
    X_test, y_test = test[feature_cols], test["target_next_day_up"]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== NEXT-DAY DIRECTION BASELINE ===")
    print(f"Training rows: {len(train)} | Test rows: {len(test)}")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"F1 score : {f1_score(y_test, predictions, zero_division=0):.3f}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    test_results = test[["date", "close", "sentiment", "target_next_day_up"]].copy()
    test_results["predicted_next_day_up"] = predictions
    test_results.to_csv(ROOT / "test_predictions.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(df["date"], df["close"], label="Close price")
    axes[0].set_title("Sample stock price (synthetic demo data)")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[1].plot(df["date"], df["sentiment"], label="Daily news sentiment")
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_title("Daily headline sentiment (VADER)")
    axes[1].set_ylabel("Compound score")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(ROOT / "price_and_sentiment.png", dpi=150)
    print("\nSaved: test_predictions.csv and price_and_sentiment.png")

if __name__ == "__main__":
    main()
