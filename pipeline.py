import json
import logging
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import config
from data import loadSplit
from environment import IndiaStockTradingEnv
from metrics import computeMetrics, equalWeightBaseline, momentumBaseline, niftyBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def makeEnv(dataFrame): return lambda: IndiaStockTradingEnv(dataFrame=dataFrame, tickers=config.tickers, initialCapital=config.initialCapital, maxShares=config.maxSharesPerStock, transactionCost=config.transactionCostRate, rewardScaling=config.rewardScaling)

def backtest(model, testData):
    tradingEnv = IndiaStockTradingEnv(dataFrame=testData, tickers=config.tickers, initialCapital=config.initialCapital, maxShares=config.maxSharesPerStock, transactionCost=config.transactionCostRate, rewardScaling=config.rewardScaling)
    observation, _ = tradingEnv.reset()
    values, terminated, truncated = [config.initialCapital], False, False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = tradingEnv.step(action)
        values.append(info["portfolioValue"])
    return values

def runAlgo(algoName, modelClass, buildKwargs, timesteps=config.timesteps):
    config.ensureDataDir()
    algoKey = algoName.lower()
    logger = logging.getLogger(algoKey)
    trainData, validationData, testData = loadSplit()
    trainEnv, validationEnv = DummyVecEnv([makeEnv(trainData)]), DummyVecEnv([makeEnv(validationData)])
    bestModelPath = config.dataDir / f"{algoKey}_best_model"
    evalCallback = EvalCallback(validationEnv, best_model_save_path=str(config.dataDir), log_path=str(config.dataDir), eval_freq=5000, n_eval_episodes=1, deterministic=True, render=False)
    model = modelClass("MlpPolicy", trainEnv, **buildKwargs(len(config.tickers)))
    logger.info("training %s for %s timesteps with FinRL features", algoKey, f"{timesteps:,}")
    model.learn(total_timesteps=timesteps, callback=evalCallback)
    savedBest = config.dataDir / "best_model.zip"
    if savedBest.exists():
        savedBest.replace(bestModelPath.with_suffix(".zip"))
        model = modelClass.load(str(bestModelPath.with_suffix(".zip")), env=trainEnv)
    values = backtest(model, testData)
    results = {algoKey: computeMetrics(values), "niftyBuyHold": computeMetrics(niftyBaseline(config.testStart, config.testEnd, config.initialCapital)), "equalWeightBuyHold": computeMetrics(equalWeightBaseline(testData, config.initialCapital)), "momentumRule": computeMetrics(momentumBaseline(testData, config.initialCapital))}
    resultPath = config.dataDir / f"{algoKey}_metrics.json"
    resultPath.write_text(json.dumps(results, indent=2), encoding="utf-8")
    modelPath = config.dataDir / f"{algoKey}_final"
    model.save(str(modelPath))
    logger.info("metrics saved -> %s", resultPath)
    logger.info("model saved -> %s.zip", modelPath)
    return results
