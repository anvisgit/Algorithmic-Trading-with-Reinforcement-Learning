import argparse
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import config
from pipeline import runAlgo

def buildKwargs(stockCount): return {"learning_rate": 3e-4, "buffer_size": 100_000, "batch_size": 256, "tau": 0.005, "gamma": 0.99, "action_noise": OrnsteinUhlenbeckActionNoise(mean=np.zeros(stockCount), sigma=0.1 * np.ones(stockCount), theta=0.15), "train_freq": (1, "episode"), "gradient_steps": -1, "learning_starts": 2_000, "policy_kwargs": {"net_arch": [256, 256, 128]}, "verbose": 0, "seed": config.seed}
def main():
    parser = argparse.ArgumentParser(description="Train TD3 with FinRL features.")
    parser.add_argument("--timesteps", type=int, default=config.timesteps)
    runAlgo("td3", TD3, buildKwargs, parser.parse_args().timesteps)
if __name__ == "__main__": main()
