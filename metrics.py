import numpy as np
import pandas as pd
import yfinance as yf
import config

def computeMetrics(portfolioSeries, riskFreeRate=config.riskFreeRate):
    portfolioValue = pd.Series(portfolioSeries).reset_index(drop=True)
    returns = portfolioValue.pct_change().dropna()
    totalReturn = portfolioValue.iloc[-1] / portfolioValue.iloc[0] - 1
    annualReturn = (1 + totalReturn) ** (252 / len(portfolioValue)) - 1
    annualVolatility = returns.std() * np.sqrt(252)
    drawdown = (portfolioValue - portfolioValue.cummax()) / (portfolioValue.cummax() + 1e-9)
    maxDrawdown = drawdown.min()
    return {"totalReturn": float(totalReturn), "annualReturn": float(annualReturn), "volatility": float(annualVolatility), "sharpe": float((annualReturn - riskFreeRate) / (annualVolatility + 1e-9)), "maxDrawdown": float(maxDrawdown), "calmar": float(annualReturn / (abs(maxDrawdown) + 1e-9)), "winRate": float((returns > 0).mean()), "finalValue": float(portfolioValue.iloc[-1])}

def niftyBaseline(startDate, endDate, initialCapital):
    niftyData = yf.download("^NSEI", start=startDate, end=endDate, auto_adjust=True, progress=False)[["Close"]].rename(columns={"Close": "close"}).dropna()
    return initialCapital * (niftyData["close"] / niftyData["close"].iloc[0])

def equalWeightBaseline(testData, initialCapital):
    averageClose = testData.groupby("date")["close"].mean()
    return initialCapital * (averageClose / averageClose.iloc[0])

def momentumBaseline(testData, initialCapital):
    baselineData = testData.copy()
    baselineData["sma20"] = baselineData.groupby("tic")["close"].transform(lambda values: values.rolling(20).mean())
    baselineData["signal"] = (baselineData["close"] > baselineData["sma20"]).astype(int)
    dailyPrice = baselineData.groupby("date").apply(lambda group: group.loc[group["signal"] == 1, "close"].mean() if group["signal"].sum() else np.nan).ffill().bfill()
    return initialCapital * (dailyPrice / dailyPrice.iloc[0])
