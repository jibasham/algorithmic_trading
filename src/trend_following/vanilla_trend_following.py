import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import yfinance as yf


def fetch_data(ticker: str, period: str = "5y"):
    data = yf.download(ticker, period=period)
    return data


def calculate_indicators(data: pd.DataFrame):
    data["EMA_50"] = ta.trend.ema_indicator(data["Close"], window=50)
    data["EMA_200"] = ta.trend.ema_indicator(data["Close"], window=200)
    data["ATR"] = ta.volatility.average_true_range(
        data["High"], data["Low"], data["Close"], window=14
    )
    return data


def generate_ema_signals(data: pd.DataFrame):
    data["EMA_Signal"] = 0
    data["EMA_Signal"][50:] = np.where(
        (data["EMA_50"][50:] > data["EMA_200"][50:])
        & (data["Close"][50:] > data["EMA_200"][50:]),
        1,
        0,
    )
    data["EMA_Signal"] = data["EMA_Signal"].diff()
    return data


def calculate_bollinger_bands(
    data: pd.DataFrame, window: int = 20, sigma1: float = 1, sigma2: float = 2
):
    data["BB_Middle"] = data["Close"].rolling(window=window).mean()
    data["BB_Std"] = data["Close"].rolling(window=window).std()
    data["BB_Upper1"] = data["BB_Middle"] + (sigma1 * data["BB_Std"])
    data["BB_Lower1"] = data["BB_Middle"] - (sigma1 * data["BB_Std"])
    data["BB_Upper2"] = data["BB_Middle"] + (sigma2 * data["BB_Std"])
    data["BB_Lower2"] = data["BB_Middle"] - (sigma2 * data["BB_Std"])
    return data


def generate_bollinger_signals(data: pd.DataFrame):
    data["BB_Signal"] = 0
    data["BB_Signal"][20:] = np.where(
        (data["Close"][20:] > data["BB_Upper1"][20:])
        & (data["Close"][20:] < data["BB_Upper2"][20:]),
        1,
        0,
    )
    data["BB_Signal"][20:] = np.where(
        (data["Close"][20:] < data["BB_Lower1"][20:])
        & (data["Close"][20:] > data["BB_Lower2"][20:]),
        -1,
        data["BB_Signal"][20:],
    )
    return data


def position_sizing(data: pd.DataFrame, initial_capital: float = 100000):
    data["Position_EMA"] = data["EMA_Signal"].apply(
        lambda x: initial_capital if x == 1 else (-initial_capital if x == -1 else 0)
    )
    data["Position_BB"] = data["BB_Signal"].apply(
        lambda x: initial_capital if x == 1 else (-initial_capital if x == -1 else 0)
    )
    return data


def plot_bollinger_bands(data: pd.DataFrame):
    plt.figure(figsize=(14, 7))
    plt.plot(data["Close"], label="QQQ Close Price")
    plt.plot(data["BB_Middle"], label="BB Middle Band", linestyle="--")
    plt.plot(data["BB_Upper1"], label="BB Upper Band (1 sigma)", linestyle="--")
    plt.plot(data["BB_Lower1"], label="BB Lower Band (1 sigma)", linestyle="--")
    plt.plot(data["BB_Upper2"], label="BB Upper Band (2 sigma)", linestyle="--")
    plt.plot(data["BB_Lower2"], label="BB Lower Band (2 sigma)", linestyle="--")
    buy_signals = data[data["Buy_Sell"] == "Buy"]
    sell_signals = data[data["Buy_Sell"] == "Sell"]
    buy_to_cover_signals = data[data["Buy_Sell"] == "Buy to Cover"]
    sell_short_signals = data[data["Buy_Sell"] == "Sell Short"]
    plt.scatter(
        buy_signals.index,
        buy_signals["Close"],
        marker="^",
        color="g",
        label="Buy",
        alpha=1,
    )
    plt.scatter(
        sell_signals.index,
        sell_signals["Close"],
        marker="v",
        color="r",
        label="Sell",
        alpha=1,
    )
    plt.scatter(
        buy_to_cover_signals.index,
        buy_to_cover_signals["Close"],
        marker="^",
        color="b",
        label="Buy to Cover",
        alpha=1,
    )
    plt.scatter(
        sell_short_signals.index,
        sell_short_signals["Close"],
        marker="v",
        color="orange",
        label="Sell Short",
        alpha=1,
    )
    plt.legend()
    plt.title("Bollinger Bands Strategy with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show(block=True)


def backtest_strategy(
    data: pd.DataFrame,
    strategy: str,
    initial_capital: float = 100000,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.1,
):
    data["Holdings"] = 0
    data["Cash"] = initial_capital
    data["Total"] = initial_capital
    data["Position"] = data[f"Position_{strategy}"]
    data["Buy_Sell"] = np.nan

    for i in range(1, len(data)):
        if data["Position"].iloc[i - 1] > 0:  # Long position
            stop_loss = data["Close"].iloc[i - 1] * (1 - stop_loss_pct)
            take_profit = data["Close"].iloc[i - 1] * (1 + take_profit_pct)
            if (
                data["Close"].iloc[i] <= stop_loss
                or data["Close"].iloc[i] >= take_profit
            ):
                data["Position"].iloc[i] = 0
                data.at[data.index[i], "Buy_Sell"] = "Sell"
            else:
                data["Position"].iloc[i] = (
                    data["Cash"].iloc[i - 1] // data["Close"].iloc[i]
                )
        elif data["Position"].iloc[i - 1] < 0:  # Short position
            stop_loss = data["Close"].iloc[i - 1] * (1 + stop_loss_pct)
            take_profit = data["Close"].iloc[i - 1] * (1 - take_profit_pct)
            if (
                data["Close"].iloc[i] >= stop_loss
                or data["Close"].iloc[i] <= take_profit
            ):
                data["Position"].iloc[i] = 0
                data.at[data.index[i], "Buy_Sell"] = "Buy to Cover"
            else:
                data["Position"].iloc[i] = -(
                    data["Cash"].iloc[i - 1] // data["Close"].iloc[i]
                )
        else:
            data["Position"].iloc[i] = data["Position"].iloc[i]
            if data["Position"].iloc[i] > 0:
                data.at[data.index[i], "Buy_Sell"] = "Buy"
            elif data["Position"].iloc[i] < 0:
                data.at[data.index[i], "Buy_Sell"] = "Sell Short"

        data["Holdings"].iloc[i] = data["Position"].iloc[i] * data["Close"].iloc[i]
        if data["Position"].iloc[i] == 0:
            data["Cash"].iloc[i] = (
                data["Cash"].iloc[i - 1] + data["Holdings"].iloc[i - 1]
            )
        else:
            data["Cash"].iloc[i] = data["Cash"].iloc[i - 1]
        data["Total"].iloc[i] = data["Holdings"].iloc[i] + data["Cash"].iloc[i]

    return data


def calculate_performance_metrics(data: pd.DataFrame):
    total_return = data["Total"].iloc[-1] / data["Total"].iloc[0] - 1
    returns = data["Total"].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    drawdown = data["Total"] / data["Total"].cummax() - 1
    max_drawdown = drawdown.min()
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }


def plot_results(data_ema: pd.DataFrame, data_bb: pd.DataFrame):
    plt.figure(figsize=(14, 7))
    plt.plot(data_ema["Total"], label="EMA Strategy Equity")
    plt.plot(data_bb["Total"], label="Bollinger Bands Strategy Equity")
    plt.plot(data_ema["Close"], label="QQQ Close Price")
    plt.legend()
    plt.title("Trend Following Strategy Performance")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show(block=True)


def plot_signals(data: pd.DataFrame, strategy: str):
    plt.figure(figsize=(14, 7))
    plt.plot(data["Close"], label="QQQ Close Price")
    buy_signals = data[data["Buy_Sell"] == "Buy"]
    sell_signals = data[data["Buy_Sell"] == "Sell"]
    buy_to_cover_signals = data[data["Buy_Sell"] == "Buy to Cover"]
    sell_short_signals = data[data["Buy_Sell"] == "Sell Short"]
    plt.scatter(
        buy_signals.index,
        buy_signals["Close"],
        marker="^",
        color="g",
        label="Buy",
        alpha=1,
    )
    plt.scatter(
        sell_signals.index,
        sell_signals["Close"],
        marker="v",
        color="r",
        label="Sell",
        alpha=1,
    )
    plt.scatter(
        buy_to_cover_signals.index,
        buy_to_cover_signals["Close"],
        marker="^",
        color="b",
        label="Buy to Cover",
        alpha=1,
    )
    plt.scatter(
        sell_short_signals.index,
        sell_short_signals["Close"],
        marker="v",
        color="orange",
        label="Sell Short",
        alpha=1,
    )
    plt.legend()
    plt.title(f"{strategy} Strategy Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show(block=True)


def main():
    # Configuration for Bollinger Bands
    bollinger_window = 20
    sigma1 = 1
    sigma2 = 2

    # Fetch QQQ data
    qqq_data = fetch_data("QQQ")

    # Calculate indicators
    qqq_data = calculate_indicators(qqq_data)
    qqq_data = calculate_bollinger_bands(
        qqq_data, window=bollinger_window, sigma1=sigma1, sigma2=sigma2
    )

    # Generate buy/sell signals
    qqq_data = generate_ema_signals(qqq_data)
    qqq_data = generate_bollinger_signals(qqq_data)

    # Position sizing
    qqq_data = position_sizing(qqq_data)

    # Backtest the EMA strategy
    qqq_data_ema = backtest_strategy(qqq_data.copy(), "EMA")
    performance_metrics_ema = calculate_performance_metrics(qqq_data_ema)
    print("EMA Strategy Performance Metrics:", performance_metrics_ema)

    # Backtest the Bollinger Bands strategy
    qqq_data_bb = backtest_strategy(qqq_data.copy(), "BB")
    performance_metrics_bb = calculate_performance_metrics(qqq_data_bb)
    print("Bollinger Bands Strategy Performance Metrics:", performance_metrics_bb)

    # Plot the results
    plot_results(qqq_data_ema, qqq_data_bb)

    # Plot the buy/sell signals for EMA strategy
    plot_signals(qqq_data_ema, "EMA")

    # Plot the buy/sell signals and Bollinger Bands for Bollinger Bands strategy
    plot_bollinger_bands(qqq_data_bb)


if __name__ == "__main__":
    main()

    plt.plot([1])
    plt.show(block=True)


if __name__ == "__main__":
    main()
