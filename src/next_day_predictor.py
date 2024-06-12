from datetime import datetime, timedelta

import pandas as pd
import requests
import ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from textblob import TextBlob
from tqdm import tqdm

NEWS_API_KEY = "6079c62780ba49d581c9ba15817f6cb1"
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# List of ticker symbols
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


def fetch_stock_data(tickers: list[str], start_date: str, end_date: str) -> dict:
    """
    Fetch historical stock price data for a list of ticker symbols.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols to fetch data for.
    start_date : str
        Start date for fetching data (YYYY-MM-DD).
    end_date : str
        End date for fetching data (YYYY-MM-DD).

    Returns
    -------
    dict
        Dictionary with ticker symbols as keys and their corresponding historical data as values.
    """
    data = {}
    for ticker in tqdm(tickers, desc="Fetching stock data"):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data
    return data


def fetch_news(ticker: str, from_date: str, to_date: str) -> dict:
    """
    Fetch news articles for a given ticker symbol within a date range.

    Parameters
    ----------
    ticker : str
        Ticker symbol to fetch news for.
    from_date : str
        Start date for fetching news (YYYY-MM-DD).
    to_date : str
        End date for fetching news (YYYY-MM-DD).

    Returns
    -------
    dict
        JSON response from the news API containing news articles.
    """
    params = {
        "q": ticker,
        "from": from_date,
        "to": to_date,
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY,
    }
    response = requests.get(NEWS_API_ENDPOINT, params=params)
    response.raise_for_status()
    return response.json()


def analyze_sentiment(article: str) -> float:
    """
    Analyze the sentiment of a news article.

    Parameters
    ----------
    article : str
        Text of the news article to analyze.

    Returns
    -------
    float
        Sentiment polarity score.
    """
    analysis = TextBlob(article)
    return analysis.sentiment.polarity


def fetch_sentiment_data(tickers: list[str], from_date: str, to_date: str) -> dict:
    """
    Fetch and analyze sentiment data for a list of ticker symbols.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols to fetch sentiment data for.
    from_date : str
        Start date for fetching sentiment data (YYYY-MM-DD).
    to_date : str
        End date for fetching sentiment data (YYYY-MM-DD).

    Returns
    -------
    dict
        Dictionary with ticker symbols as keys and their corresponding average sentiment and individual sentiments as values.
    """
    sentiment_data = {}
    for ticker in tqdm(tickers, desc="Fetching sentiment data"):
        articles = fetch_news(ticker, from_date, to_date)
        sentiments = []
        article_count = len(articles["articles"])
        for article in articles["articles"]:
            sentiment = analyze_sentiment(article["description"] or article["title"])
            sentiments.append(sentiment)
        sentiment_data[ticker] = {
            "average_sentiment": sum(sentiments) / len(sentiments) if sentiments else 0,
            "sentiments": sentiments,
            "article_count": article_count,
        }
    return sentiment_data


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given DataFrame of stock data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stock data.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for technical indicators.
    """
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["RSI"] = ta.momentum.rsi(df["Close"])
    df["MACD"] = ta.trend.macd(df["Close"])
    df["Bollinger_High"] = ta.volatility.bollinger_hband(df["Close"])
    df["Bollinger_Low"] = ta.volatility.bollinger_lband(df["Close"])
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["A/D"] = ta.volume.acc_dist_index(
        df["High"], df["Low"], df["Close"], df["Volume"]
    )
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"])
    df["Aroon_Up"] = ta.trend.aroon_up(df["Close"])
    df["Aroon_Down"] = ta.trend.aroon_down(df["Close"])
    df["Stoch_Oscillator"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
    return df


def preprocess_data(stock_data: dict, sentiment_data: dict) -> dict:
    """
    Preprocess and merge stock data with sentiment data.

    Parameters
    ----------
    stock_data : dict
        Dictionary with ticker symbols as keys and their corresponding historical stock data as values.
    sentiment_data : dict
        Dictionary with ticker symbols as keys and their corresponding sentiment data as values.

    Returns
    -------
    dict
        Dictionary with ticker symbols as keys and their corresponding preprocessed data as values.
    """
    processed_data = {}
    for ticker in tqdm(stock_data, desc="Preprocessing data"):
        df = stock_data[ticker].copy()
        df["Sentiment"] = sentiment_data[ticker]["average_sentiment"]
        df["Article_Count"] = sentiment_data[ticker]["article_count"]
        df = calculate_technical_indicators(df)
        processed_data[ticker] = df
    return processed_data


def prepare_data(processed_data: dict) -> pd.DataFrame:
    """
    Prepare the data for training and testing.

    Parameters
    ----------
    processed_data : dict
        Dictionary with ticker symbols as keys and their corresponding preprocessed data as values.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all ticker data, with target column for model training.
    """
    df_list = []
    for ticker, data in tqdm(processed_data.items(), desc="Preparing data"):
        data["Ticker"] = ticker
        data["Target"] = data["Close"].shift(-1) > data["Close"]
        data.dropna(inplace=True)
        df_list.append(data)
    return pd.concat(df_list)


def optimize_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Perform grid search for hyperparameter tuning of the Random Forest model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target set.

    Returns
    -------
    RandomForestClassifier
        Best estimator from grid search.
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [10, 20, 30, None],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def walk_forward_validation(
    data: pd.DataFrame, initial_train_size: int, step_size: int
) -> dict:
    """
    Perform walk-forward validation on the data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the full dataset for walk-forward validation.
    initial_train_size : int
        Size of the initial training set.
    step_size : int
        Step size for walk-forward validation.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, and recall for each step.
    """
    results = {"accuracy": [], "precision": [], "recall": []}
    for start in range(0, len(data) - initial_train_size, step_size):
        train = data.iloc[start : start + initial_train_size]
        test = data.iloc[
            start + initial_train_size : start + initial_train_size + step_size
        ]

        X_train, y_train = train.drop(columns=["Target", "Ticker"]), train["Target"]
        X_test, y_test = test.drop(columns=["Target", "Ticker"]), test["Target"]

        model = optimize_model(X_train, y_train)
        y_pred = model.predict(X_test)

        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["precision"].append(precision_score(y_test, y_pred))
        results["recall"].append(recall_score(y_test, y_pred))

    return results


def save_predictions_to_markdown(predictions: dict, filename: str):
    """
    Save daily prediction results to a markdown file.

    Parameters
    ----------
    predictions : dict
        Dictionary with ticker symbols as keys and their corresponding predictions as values.
    filename : str
        Filename for the markdown file.

    Returns
    -------
    None
    """
    with open(filename, "w") as file:
        file.write("# Daily Stock Predictions\n\n")
        file.write("| Ticker | Prediction |\n")
        file.write("|--------|-------------|\n")
        for ticker, prediction in predictions.items():
            file.write(f"| {ticker} | {'Up' if prediction else 'Down'} |\n")


def main():
    """
    Main function to orchestrate data acquisition, integration, backtesting, and daily prediction.

    Parameters
    -------
    None

    Returns
    -------
    None
    """
    # Define the time range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)  # One year of data

    print("Starting data acquisition...")
    # Fetch stock price data
    stock_data = fetch_stock_data(
        tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )

    print("Fetching sentiment data...")
    # Fetch sentiment data for the last 7 days
    sentiment_data = fetch_sentiment_data(
        tickers,
        (end_date - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    print("Preprocessing data...")
    # Preprocess and integrate data
    processed_data = preprocess_data(stock_data, sentiment_data)

    print("Preparing data for walk-forward validation...")
    # Prepare data for walk-forward validation
    df = prepare_data(processed_data)

    print("Starting walk-forward validation...")
    # Walk-forward validation
    initial_train_size = int(0.6 * len(df))
    step_size = int(0.1 * len(df))
    walk_forward_results = walk_forward_validation(df, initial_train_size, step_size)

    print("Walk-forward validation results:")
    # Display walk-forward validation results
    for metric, values in walk_forward_results.items():
        print(f"{metric.capitalize()}: {sum(values)/len(values):.2f}")

    print("Making daily predictions...")
    # Predict stock movement for the next day
    predictions = {}
    for ticker in tickers:
        latest_data = (
            processed_data[ticker].iloc[-1:].drop(columns=["Target", "Ticker"])
        )
        model = optimize_model(latest_data, processed_data[ticker]["Target"])
        prediction = model.predict(latest_data)
        predictions[ticker] = prediction[0]

    print("Saving predictions to markdown...")
    # Save predictions to markdown
    save_predictions_to_markdown(predictions, "daily_predictions.md")

    print("Process completed.")


if __name__ == "__main__":
    main()
