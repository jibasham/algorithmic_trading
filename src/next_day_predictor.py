import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
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
    filename = Path(f".cache/news_{ticker}_{from_date}_{to_date}.json")

    # if file exists, read and return
    if filename.exists():
        with filename.open("r") as file:
            print(f"News data loaded from {filename}")
            return json.load(file)

    response = requests.get(NEWS_API_ENDPOINT, params=params)
    response.raise_for_status()

    # Ensure directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)

    # write response to file
    with filename.open("w") as file:
        json.dump(response.json(), file)
    print(f"News data saved to {filename}")

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
    Fetch and analyze sentiment data for a list of ticker symbols weekly, then unpack to daily.

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
        Dictionary with ticker symbols as keys and their corresponding daily sentiment data as values.
    """
    sentiment_data = {ticker: [] for ticker in tickers}
    current_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")

    while current_date <= end_date:
        week_start = current_date
        week_end = current_date + timedelta(days=6)

        for ticker in tickers:
            articles = fetch_news(
                ticker, week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")
            )
            daily_sentiments = {}

            for article in articles["articles"]:
                publish_date = article["publishedAt"][:10]
                sentiment = analyze_sentiment(
                    article["description"] or article["title"]
                )

                if publish_date not in daily_sentiments:
                    daily_sentiments[publish_date] = []
                daily_sentiments[publish_date].append(sentiment)

            # Calculate daily average sentiments and weekly average
            daily_avg_sentiments = {
                date: sum(sentiments) / len(sentiments)
                for date, sentiments in daily_sentiments.items()
            }
            weekly_avg_sentiment = (
                sum(daily_avg_sentiments.values()) / len(daily_avg_sentiments)
                if daily_avg_sentiments
                else 0
            )

            for single_date in (week_start + timedelta(n) for n in range(7)):
                date_str = single_date.strftime("%Y-%m-%d")
                sentiment = daily_avg_sentiments.get(date_str, weekly_avg_sentiment)
                article_count = len(daily_sentiments.get(date_str, []))

                sentiment_data[ticker].append(
                    {
                        "date": date_str,
                        "average_sentiment": sentiment,
                        "article_count": article_count,
                    }
                )

        current_date += timedelta(days=7)

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
    df["Aroon_Up"] = ta.trend.aroon_up(df["High"], df["Low"])
    df["Aroon_Down"] = ta.trend.aroon_down(df["High"], df["Low"])
    df["Stoch_Oscillator"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
    df["CCI"] = ta.trend.cci(df["High"], df["Low"], df["Close"])
    df["ROC"] = ta.momentum.roc(df["Close"])
    return df


def preprocess_data(stock_data: dict, sentiment_data: dict) -> dict:
    """
    Preprocess and merge stock data with daily sentiment data.

    Parameters
    ----------
    stock_data : dict
        Dictionary with ticker symbols as keys and their corresponding historical stock data as values.
    sentiment_data : dict
        Dictionary with ticker symbols as keys and their corresponding daily sentiment data as values.

    Returns
    -------
    dict
        Dictionary with ticker symbols as keys and their corresponding preprocessed data as values.
    """
    processed_data = {}
    for ticker in tqdm(stock_data, desc="Preprocessing data"):
        df = stock_data[ticker].copy()
        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data[ticker])
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
            df = df.merge(sentiment_df, left_index=True, right_on="date", how="left")
            df.set_index("date", inplace=True)
            df["Sentiment"] = df["average_sentiment"]
            df["Article_Count"] = df["article_count"]
            df = df.drop(columns=["average_sentiment", "article_count"])
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
        "max_features": ["sqrt", "log2"],
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


def optimize_xgboost_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> xgb.XGBClassifier:
    """
    Perform grid search for hyperparameter tuning of the XGBoost model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target set.

    Returns
    -------
    xgb.XGBClassifier
        Best estimator from grid search.
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    xgb_model = xgb.XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    )
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def save_dataframe_to_parquet(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame to a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Filename for the Parquet file.

    Returns
    -------
    None
    """
    df.to_parquet(filename)


def save_model(model, filename: str):
    """
    Save a model to a file.

    Parameters
    ----------
    model :
        Model to save.
    filename : str
        Filename for the model file.

    Returns
    -------
    None
    """
    joblib.dump(model, filename)


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
    start_date = end_date - timedelta(days=20 * 365)  # 20 years of data
    sentiment_start_date = end_date - timedelta(days=30)  # Free tier limit

    print("Starting data acquisition...")
    # Fetch stock price data
    stock_data = fetch_stock_data(
        tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )

    use_sentiment_data = (
        False  # Set this flag to True to enable sentiment data processing
    )

    if use_sentiment_data:
        print("Fetching sentiment data...")
        # Fetch sentiment data for the last 30 days
        sentiment_data = fetch_sentiment_data(
            tickers,
            sentiment_start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
    else:
        sentiment_data = {}

    print("Preprocessing data...")
    # Preprocess and integrate data
    processed_data = preprocess_data(stock_data, sentiment_data)

    print("Preparing data for walk-forward validation...")
    # Prepare data for walk-forward validation
    df = prepare_data(processed_data)
    save_dataframe_to_parquet(df, "prepared_data.parquet")

    print("Starting walk-forward validation...")
    # Walk-forward validation
    initial_train_size = int(0.6 * len(df))
    step_size = int(0.1 * len(df))
    walk_forward_results_rf = walk_forward_validation(
        df, initial_train_size, step_size, optimize_model
    )
    walk_forward_results_xgb = walk_forward_validation(
        df, initial_train_size, step_size, optimize_xgboost_model
    )

    print("Walk-forward validation results for Random Forest:")
    for metric, values in walk_forward_results_rf.items():
        print(f"{metric.capitalize()}: {sum(values)/len(values):.2f}")

    print("Walk-forward validation results for XGBoost:")
    for metric, values in walk_forward_results_xgb.items():
        print(f"{metric.capitalize()}: {sum(values)/len(values):.2f}")

    print("Making daily predictions...")
    # Predict stock movement for the next day using both models
    predictions_rf = {}
    predictions_xgb = {}
    models_rf = {}
    models_xgb = {}

    latest_data = df.iloc[-30:]
    X_latest = latest_data.drop(columns=["Target", "Ticker"])
    y_latest = latest_data["Target"]

    model_rf = optimize_model(X_latest, y_latest)
    predictions_rf = model_rf.predict(X_latest.tail(len(tickers)))
    models_rf = model_rf

    model_xgb = optimize_xgboost_model(X_latest, y_latest)
    predictions_xgb = model_xgb.predict(X_latest.tail(len(tickers)))
    models_xgb = model_xgb

    print("Saving predictions to markdown...")
    save_predictions_to_markdown(
        dict(zip(tickers, predictions_rf)), "daily_predictions_rf.md"
    )
    save_predictions_to_markdown(
        dict(zip(tickers, predictions_xgb)), "daily_predictions_xgb.md"
    )

    print("Saving models...")
    save_model(models_rf, "model_rf.joblib")
    save_model(models_xgb, "model_xgb.joblib")

    print("Process completed.")


if __name__ == "__main__":
    main()
