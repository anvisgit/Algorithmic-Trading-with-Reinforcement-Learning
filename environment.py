import gymnasium as gym
import numpy as np
from gymnasium import spaces

class IndiaStockTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, dataFrame, tickers, initialCapital, maxShares, transactionCost, rewardScaling):
        super().__init__()
        self.dataFrame = dataFrame.copy()
        self.tickers = tickers
        self.stockCount = len(tickers)
        self.initialCapital = initialCapital
        self.maxShares = maxShares
        self.transactionCost = transactionCost
        self.rewardScaling = rewardScaling
        self.featureColumns = [column for column in dataFrame.columns if column not in ("date", "tic", "open", "high", "low")]
        self.featureCount = len(self.featureColumns)
        self.dates = sorted(dataFrame["date"].unique())
        self.dayCount = len(self.dates)
        self.maxTradePerStep = max(1, int(maxShares * 0.25))
        observationSize = 1 + self.stockCount + self.stockCount * self.featureCount
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stockCount,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observationSize,), dtype=np.float32)
    def dayRows(self, date): return self.dataFrame[self.dataFrame["date"] == date]
    def prices(self, date):
        rows = self.dayRows(date)
        return {ticker: float(match["close"].iloc[0]) if len(match := rows[rows["tic"] == ticker]) else 0.0 for ticker in self.tickers}
    def features(self, date):
        rows, blocks = self.dayRows(date), []
        for ticker in self.tickers:
            tickerRows = rows[rows["tic"] == ticker]
            blocks.append(np.clip(tickerRows[self.featureColumns].iloc[0].values.astype(np.float32), -10, 10) if len(tickerRows) else np.zeros(self.featureCount, dtype=np.float32))
        return np.concatenate(blocks)
    def portfolioValue(self, prices): return self.cash + sum(self.holdings[ticker] * prices[ticker] for ticker in self.tickers)
    def observation(self):
        prices = self.prices(self.dates[self.stepIndex])
        value = self.portfolioValue(prices)
        cashRatio = self.cash / (value + 1e-9)
        holdingRatios = np.array([self.holdings[ticker] * prices[ticker] / (value + 1e-9) for ticker in self.tickers], dtype=np.float32)
        return np.concatenate([[cashRatio], holdingRatios, np.nan_to_num(self.features(self.dates[self.stepIndex]), nan=0.0, posinf=1.0, neginf=-1.0)]).astype(np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.stepIndex = 0
        self.cash = float(self.initialCapital)
        self.holdings = {ticker: 0 for ticker in self.tickers}
        self.lastAction = np.zeros(self.stockCount, dtype=np.float32)
        self.peakPortfolio = float(self.initialCapital)
        return self.observation(), {}
    def step(self, action):
        currentPrices = self.prices(self.dates[self.stepIndex])
        valueBefore = self.portfolioValue(currentPrices)
        totalCost = 0.0
        clippedAction = np.clip(np.array(action, dtype=np.float32), -1, 1)
        targetShares = np.round((clippedAction + 1.0) / 2.0 * self.maxShares).astype(int)
        for index, ticker in enumerate(self.tickers):
            price = currentPrices[ticker]
            if price <= 0: continue
            desiredShares = int(np.clip(targetShares[index], 0, self.maxShares))
            shareDelta = int(np.clip(desiredShares - self.holdings[ticker], -self.maxTradePerStep, self.maxTradePerStep))
            if shareDelta > 0:
                buyShares = min(shareDelta, int(self.cash / (price * (1 + self.transactionCost))))
                buyCost = buyShares * price * (1 + self.transactionCost)
                if buyShares > 0 and buyCost <= self.cash:
                    self.cash -= buyCost
                    self.holdings[ticker] += buyShares
                    totalCost += buyShares * price * self.transactionCost
            elif shareDelta < 0:
                sellShares = min(abs(shareDelta), self.holdings[ticker])
                if sellShares > 0:
                    self.cash += sellShares * price * (1 - self.transactionCost)
                    self.holdings[ticker] -= sellShares
                    totalCost += sellShares * price * self.transactionCost
        self.stepIndex += 1
        terminated = self.stepIndex >= self.dayCount - 1
        valueAfter = self.portfolioValue(self.prices(self.dates[self.stepIndex]))
        drawdown = max(0.0, (self.peakPortfolio - valueAfter) / (self.peakPortfolio + 1e-9))
        actionChange = np.sum(np.abs(clippedAction - self.lastAction))
        self.peakPortfolio = max(self.peakPortfolio, valueAfter)
        self.lastAction = clippedAction
        reward = (valueAfter - valueBefore) * self.rewardScaling - totalCost * self.rewardScaling * 0.8 - actionChange * self.rewardScaling * 0.15 - drawdown * 0.1
        return self.observation(), float(reward), terminated, False, {"portfolioValue": valueAfter, "cash": self.cash, "transactionCost": totalCost, "drawdown": drawdown}
