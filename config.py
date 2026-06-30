from pathlib import Path

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
trainStart = "2018-01-01"
trainEnd = "2022-12-31"
testStart = "2023-01-01"
testEnd = "2024-12-31"
validationFraction = 0.2
initialCapital = 1_000_000
maxSharesPerStock = 50
rewardScaling = 1e-3
sttRate = 0.001
brokerageRate = 0.0003
gstRate = 0.18
slippageRate = 0.001
transactionCostRate = sttRate + brokerageRate * (1 + gstRate) + slippageRate
indicators = ["macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
timesteps = 100_000
riskFreeRate = 0.065
seed = 42
rootDir = Path(__file__).resolve().parent
dataDir = rootDir / "data"

def ensureDataDir(): dataDir.mkdir(parents=True, exist_ok=True)
