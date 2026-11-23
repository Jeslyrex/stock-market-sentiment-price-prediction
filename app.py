
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Market Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 15px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: #171B26;
        border: 1px solid #303747;
        padding: 18px;
        border-radius: 12px;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    prices = pd.read_csv(DATA_DIR / "sample_stock_prices.csv")
    news = pd.read_csv(DATA_DIR / "sample_news_headlines.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")

    prices["date"] = pd.to_datetime(prices["date"])
    news["date"] = pd.to_datetime(news["date"])

    return prices, news, predictions, comparison


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def find_column(data, possible_names):

    for name in possible_names:
        for column in data.columns:
            if column.lower().replace(" ", "_") == name:
                return column

    return None


def calculate_sentiment(news):

    if SentimentIntensityAnalyzer is None:
        return None

    analyzer = SentimentIntensityAnalyzer()

    news = news.copy()

    news["sentiment_score"] = news["headline"].fillna("").apply(
        lambda headline: analyzer.polarity_scores(headline)["compound"]
    )

    news["sentiment"] = news["sentiment_score"].apply(
        lambda score:
        "Positive" if score >= 0.05
        else "Negative" if score <= -0.05
        else "Neutral"
    )

    return news


# --------------------------------------------------
# LOAD APPLICATION
# --------------------------------------------------

try:

    prices, news, predictions, comparison = load_data()

    news_sentiment = calculate_sentiment(news)

    # Sidebar
    st.sidebar.title("📈 Stock Analytics")
    st.sidebar.caption("Sentiment & Price Prediction")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Sentiment Analysis",
            "Model Performance",
            "Prediction Results",
            "Project Information"
        ]
    )

    st.sidebar.divider()
    st.sidebar.caption("Portfolio Project | 2025 Sample Dataset")

    # Header
    st.markdown(
        '<p class="main-title">Stock Market Sentiment & Price Prediction</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Explore stock trends, news sentiment and machine learning results.</p>',
        unsafe_allow_html=True
    )

    st.info(
        "Demonstration project using synthetic sample data. "
        "No live market data is used. Results are experimental "
        "and should not be used for investment decisions."
    )

    # --------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------

    if page == "Overview":

        st.subheader("Market Overview")

        prices = prices.sort_values("date")
        prices["daily_return"] = prices["close"].pct_change() * 100

        latest_price = prices["close"].iloc[-1]
        average_price = prices["close"].mean()
        average_volume = prices["volume"].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Stock Records", f"{len(prices):,}")
        col2.metric("Latest Closing Price", f"{latest_price:.2f}")
        col3.metric("Average Closing Price", f"{average_price:.2f}")
        col4.metric("Average Volume", f"{average_volume:,.0f}")

        st.divider()

        # Price trend
        st.subheader("Stock Price Trend")

        fig_price = px.line(
            prices,
            x="date",
            y="close",
            title="Historical Closing Price",
            labels={
                "date": "Date",
                "close": "Closing Price"
            }
        )

        fig_price.update_layout(hovermode="x unified")

        st.plotly_chart(fig_price, width="stretch")

        # Volume chart
        st.subheader("Trading Volume")

        fig_volume = px.bar(
            prices,
            x="date",
            y="volume",
            labels={
                "date": "Date",
                "volume": "Trading Volume"
            }
        )

        st.plotly_chart(fig_volume, width="stretch")

        # Returns
        st.subheader("Daily Returns")

        fig_returns = px.line(
            prices,
            x="date",
            y="daily_return",
            labels={
                "date": "Date",
                "daily_return": "Daily Return (%)"
            }
        )

        st.plotly_chart(fig_returns, width="stretch")

    # --------------------------------------------------
    # SENTIMENT ANALYSIS
    # --------------------------------------------------

    elif page == "Sentiment Analysis":

        st.subheader("News Sentiment Analysis")

        if news_sentiment is None:

            st.warning(
                "Install vaderSentiment to display sentiment analysis."
            )

        else:

            positive = (news_sentiment["sentiment"] == "Positive").sum()
            negative = (news_sentiment["sentiment"] == "Negative").sum()
            neutral = (news_sentiment["sentiment"] == "Neutral").sum()

            col1, col2, col3 = st.columns(3)

            col1.metric("Positive Headlines", positive)
            col2.metric("Negative Headlines", negative)
            col3.metric("Neutral Headlines", neutral)

            st.divider()

            sentiment_counts = (
                news_sentiment["sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_sentiment = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.45,
                title="Sentiment Distribution"
            )

            st.plotly_chart(fig_sentiment, width="stretch")

            # Sentiment score over time
            daily_sentiment = (
                news_sentiment.groupby("date")["sentiment_score"]
                .mean()
                .reset_index()
            )

            fig_daily = px.line(
                daily_sentiment,
                x="date",
                y="sentiment_score",
                title="Daily Average News Sentiment",
                labels={
                    "date": "Date",
                    "sentiment_score": "Sentiment Score"
                }
            )

            st.plotly_chart(fig_daily, width="stretch")

            st.subheader("News Headlines")

            st.dataframe(
                news_sentiment.sort_values("date", ascending=False),
                width="stretch"
            )

            st.download_button(
                "Download Sentiment Analysis",
                data=news_sentiment.to_csv(index=False),
                file_name="news_sentiment_analysis.csv",
                mime="text/csv"
            )

    # --------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------

    elif page == "Model Performance":

        st.subheader("Machine Learning Model Evaluation")

        st.write(
            "Compare the performance of the baseline, "
            "Logistic Regression and Random Forest models."
        )

        st.dataframe(comparison, width="stretch")

        metric_columns = [
            column for column in
            ["accuracy", "precision", "recall", "f1", "macro_f1"]
            if column in comparison.columns
        ]

        if metric_columns:

            selected_metric = st.selectbox(
                "Choose evaluation metric",
                metric_columns
            )

            fig_comparison = px.bar(
                comparison,
                x="model",
                y=selected_metric,
                title=f"Model Comparison: {selected_metric.title()}",
                text_auto=".3f"
            )

            fig_comparison.update_layout(
                yaxis_title=selected_metric.title(),
                xaxis_title="Machine Learning Model"
            )

            st.plotly_chart(fig_comparison, width="stretch")

        confusion_path = OUTPUT_DIR / "confusion_matrices.png"

        if confusion_path.exists():

            st.subheader("Confusion Matrices")
            st.image(str(confusion_path), width="stretch")

        st.download_button(
            "Download Model Comparison",
            data=comparison.to_csv(index=False),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    # --------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------

    elif page == "Prediction Results":

        st.subheader("Test Dataset Predictions")

        st.write(
            "These are model predictions on the held-out test dataset."
        )

        st.metric("Total Test Predictions", len(predictions))

        st.dataframe(
            predictions,
            width="stretch"
        )

        st.download_button(
            "Download Prediction Results",
            data=predictions.to_csv(index=False),
            file_name="test_predictions.csv",
            mime="text/csv"
        )

        st.caption(
            "Predictions are experimental results based on synthetic data."
        )

    # --------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------

    elif page == "Project Information":

        st.subheader("About This Project")

        st.markdown("""
        ### Project Objective

        To explore the relationship between stock price movements
        and financial news sentiment using data analytics and
        machine learning.

        ### Technologies Used

        - Python
        - Pandas and NumPy
        - VADER Sentiment Analysis
        - Scikit-learn
        - Plotly
        - Streamlit

        ### Machine Learning Models

        - Majority Class Baseline
        - Logistic Regression
        - Random Forest

        ### Project Workflow

        1. Data collection using sample datasets
        2. Data cleaning and preprocessing
        3. News sentiment analysis
        4. Feature engineering
        5. Model training and evaluation
        6. Interactive dashboard development

        ### Important Limitation

        The supplied datasets are synthetic demonstration data.
        They contain repeated sample headlines and do not represent
        actual market activity. Model performance cannot be
        generalized to real-world stock markets.
        """)

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        col1.metric("Stock Dataset Rows", len(prices))
        col2.metric("News Dataset Rows", len(news))

        st.write("Stock Dataset Preview")
        st.dataframe(prices.head(10), width="stretch")

        st.write("News Dataset Preview")
        st.dataframe(news.head(10), width="stretch")


except Exception as error:

    st.error("Dashboard could not load the required files.")

    st.exception(error)

    st.write(
        "Check that the data and outputs folders contain "
        "the required CSV files."
    )
