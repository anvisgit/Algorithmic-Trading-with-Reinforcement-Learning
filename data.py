import pandas as pd
import yfinance as yf
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
import config

def downloadData(tickers, startDate, endDate):
    rawData = yf.download(tickers, start=startDate, end=endDate, group_by="ticker", auto_adjust=True, progress=False)
    frames = []
    for ticker in tickers:
        tickerFrame = rawData[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
        tickerFrame.columns = ["open", "high", "low", "close", "volume"]
        tickerFrame["tic"] = ticker
        tickerFrame.index.name = "date"
        frames.append(tickerFrame.dropna())
    marketData = pd.concat(frames).reset_index()
    marketData["date"] = pd.to_datetime(marketData["date"])
    return marketData.sort_values(["date", "tic"]).reset_index(drop=True)

def addFinrlFeatures(marketData):
    featureEngineer = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=config.indicators, use_vix=False, use_turbulence=False, user_defined_feature=False)
    featuredData = featureEngineer.preprocess_data(marketData)
    groupedClose = featuredData.groupby("tic")["close"]
    featuredData["return1"] = groupedClose.pct_change()
    featuredData["return5"] = groupedClose.pct_change(5)
    featuredData["vol10"] = groupedClose.pct_change().rolling(10).std().reset_index(level=0, drop=True)
    featuredData["vol20"] = groupedClose.pct_change().rolling(20).std().reset_index(level=0, drop=True)
    featuredData["rangePct"] = (featuredData["high"] - featuredData["low"]) / (featuredData["low"] + 1e-9)
    featuredData["closeVsSma20"] = (featuredData["close"] - featuredData["close_30_sma"]) / (featuredData["close_30_sma"] + 1e-9)
    return featuredData.dropna().reset_index(drop=True)

def loadSplit(tickers=config.tickers):
    trainRaw = downloadData(tickers, config.trainStart, config.trainEnd)
    testRaw = downloadData(tickers, config.testStart, config.testEnd)
    trainData = addFinrlFeatures(trainRaw)
    testData = addFinrlFeatures(testRaw)
    trainDates = sorted(trainData["date"].unique())
    splitIndex = int(len(trainDates) * (1 - config.validationFraction))
    validationStart = trainDates[splitIndex]
    return trainData[trainData["date"] < validationStart].copy(), trainData[trainData["date"] >= validationStart].copy(), testData
