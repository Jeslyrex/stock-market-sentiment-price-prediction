
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_FILE = os.path.join(DATA_DIR, "sample_stock_prices.csv")
NEWS_FILE = os.path.join(DATA_DIR, "sample_news_headlines.csv")

# ==================================================
# 2. LOAD DATA
# ==================================================

print("\n========== STOCK MARKET ANALYSIS ==========\n")

prices = pd.read_csv(PRICE_FILE)
news = pd.read_csv(NEWS_FILE)

prices.columns = prices.columns.str.strip().str.lower()
news.columns = news.columns.str.strip().str.lower()

required_price = {"date", "close"}
required_news = {"date", "headline"}

if not required_price.issubset(prices.columns):
    raise ValueError("Stock CSV must contain date and close columns.")

if not required_news.issubset(news.columns):
    raise ValueError("News CSV must contain date and headline columns.")

prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
news["date"] = pd.to_datetime(news["date"], errors="coerce")

prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

prices = prices.dropna(subset=["date", "close"]).copy()
news = news.dropna(subset=["date", "headline"]).copy()

prices["date"] = prices["date"].dt.normalize()
news["date"] = news["date"].dt.normalize()

prices = (
    prices.sort_values("date")
    .drop_duplicates(subset="date")
    .reset_index(drop=True)
)

news = news.sort_values("date")

print("Stock records:", len(prices))
print("News records:", len(news))

# ==================================================
# 3. NEWS SENTIMENT
# ==================================================

print("\nCalculating news sentiment...")

analyzer = SentimentIntensityAnalyzer()

news["sentiment"] = news["headline"].astype(str).apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

daily_sentiment = (
    news.groupby("date")["sentiment"]
    .mean()
    .reset_index()
)

df = prices.merge(
    daily_sentiment,
    on="date",
    how="left"
)

df["sentiment"] = df["sentiment"].fillna(0)

# Use previous day's sentiment.
# This is a conservative timing assumption for prediction.
df["sentiment_lag_1"] = df["sentiment"].shift(1)

# ==================================================
# 4. FEATURE ENGINEERING
# ==================================================

print("Creating technical indicators...")

df["daily_return"] = df["close"].pct_change()

df["return_lag_1"] = df["daily_return"].shift(1)
df["return_lag_2"] = df["daily_return"].shift(2)
df["return_lag_3"] = df["daily_return"].shift(3)

df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()

df["price_vs_ma5"] = df["close"] / df["ma_5"] - 1
df["price_vs_ma10"] = df["close"] / df["ma_10"] - 1

# RSI
delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)

df["rsi"] = 100 - (100 / (1 + rs))
df["rsi"] = df["rsi"].fillna(50)

# MACD normalized by closing price
ema_12 = df["close"].ewm(span=12, adjust=False).mean()
ema_26 = df["close"].ewm(span=26, adjust=False).mean()

df["macd"] = (ema_12 - ema_26) / df["close"]

# ==================================================
# 5. TARGET VARIABLE
# ==================================================

# 1 = next observation's closing price is higher
# 0 = next observation's closing price is lower or equal

next_close = df["close"].shift(-1)

df["target"] = np.where(
    next_close.notna(),
    (next_close > df["close"]).astype(int),
    np.nan
)

FEATURES = [
    "daily_return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "price_vs_ma5",
    "price_vs_ma10",
    "rsi",
    "macd",
    "sentiment_lag_1"
]

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=FEATURES + ["target"]
).copy()

df["target"] = df["target"].astype(int)

X = df[FEATURES]
y = df["target"]

print("\n========== DATA SUMMARY ==========")
print("Usable records:", len(df))

print("\nTarget distribution:")
print(y.value_counts().rename(index={0: "DOWN", 1: "UP"}))

# ==================================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==================================================

split = int(len(df) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

if len(X_test) == 0 or y_train.nunique() < 2:
    raise ValueError("Insufficient data for model training and testing.")

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts().to_dict())

print("\nTesting distribution:")
print(y_test.value_counts().to_dict())

# ==================================================
# 7. DEFINE MODELS
# ==================================================

models = {
    "Majority Baseline": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.1,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )
}

# ==================================================
# 8. TIME-SERIES CROSS-VALIDATION
# ==================================================

print("\n========== TIME-SERIES CROSS-VALIDATION ==========")

cv_splits = min(5, len(X_train) // 10)

if cv_splits >= 2:
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    cv_results = []

    for name, model in models.items():

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=tscv,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro"
            }
        )

        mean_accuracy = scores["test_accuracy"].mean()
        mean_f1 = scores["test_f1_macro"].mean()

        cv_results.append({
            "Model": name,
            "CV Accuracy": mean_accuracy,
            "CV Macro F1": mean_f1
        })

        print(f"\n{name}")
        print("Mean CV Accuracy:", round(mean_accuracy, 4))
        print("Mean CV Macro F1:", round(mean_f1, 4))

    cv_df = pd.DataFrame(cv_results)

    cv_df.to_csv(
        os.path.join(OUTPUT_DIR, "cross_validation_results.csv"),
        index=False
    )

else:
    print("Not enough training data for multiple CV splits.")

# ==================================================
# 9. FINAL TEST EVALUATION
# ==================================================

print("\n========== FINAL TEST EVALUATION ==========")

results = []
predictions = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )
    }

    results.append(metrics)

    print("\nActual labels:", y_test.value_counts().to_dict())
    print("Predicted labels:", pd.Series(y_pred).value_counts().to_dict())

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["DOWN", "UP"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# ==================================================
# 10. EXPORT MODEL RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# ==================================================
# 11. EXPORT PREDICTIONS
# ==================================================

prediction_df = df.iloc[split:][
    ["date", "close", "target"]
].copy()

prediction_df = prediction_df.rename(columns={
    "close": "closing_price",
    "target": "actual_direction"
})

for name, pred in predictions.items():
    column = name.lower().replace(" ", "_")
    prediction_df[column + "_prediction"] = pred

prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False
)

# ==================================================
# 12. MODEL COMPARISON CHART
# ==================================================

plot_metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

ax = results_df.set_index("Model")[plot_metrics].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1)
)

ax.set_title("Stock Market Model Performance")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"),
    dpi=300
)

plt.close()

# ==================================================
# 13. CONFUSION MATRIX CHARTS
# ==================================================

fig, axes = plt.subplots(1, len(models), figsize=(15, 4))

for ax, (name, pred) in zip(axes, predictions.items()):

    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["DOWN", "UP"]
    )

    display.plot(ax=ax, colorbar=False)
    ax.set_title(name)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
    dpi=300
)

plt.close()

# ==================================================
# 14. STOCK PRICE AND SENTIMENT CHART
# ==================================================

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(
    df["date"],
    df["close"],
    label="Closing Price"
)

ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price")
ax1.set_title("Stock Price and News Sentiment")

ax2 = ax1.twinx()

ax2.plot(
    df["date"],
    df["sentiment"],
    label="News Sentiment",
    alpha=0.5
)

ax2.set_ylabel("Sentiment Score")

fig.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "price_and_sentiment.png"),
    dpi=300
)

plt.close()

# ==================================================
# 15. COMPLETION SUMMARY
# ==================================================

print("\n========== PROJECT COMPLETED ==========")

print("\nGenerated files inside outputs/:")

for filename in [
    "model_comparison.csv",
    "test_predictions.csv",
    "cross_validation_results.csv",
    "model_comparison.png",
    "confusion_matrices.png",
    "price_and_sentiment.png"
]:
    print("-", filename)

print("\nIMPORTANT:")
print("This project uses synthetic demonstration data.")
print("Results are experimental, not live-market predictions.")

print("\n========================================")
