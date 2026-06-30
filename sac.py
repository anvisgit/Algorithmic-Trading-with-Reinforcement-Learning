import argparse
from stable_baselines3 import SAC
import config
from pipeline import runAlgo

def buildKwargs(stockCount): return {"learning_rate": 3e-4, "buffer_size": 100_000, "batch_size": 256, "tau": 0.005, "gamma": 0.99, "ent_coef": "auto_0.1", "target_entropy": "auto", "train_freq": 4, "gradient_steps": 2, "learning_starts": 2_000, "policy_kwargs": {"net_arch": [256, 256, 128]}, "verbose": 0, "seed": config.seed}
def main():
    parser = argparse.ArgumentParser(description="Train SAC with FinRL features.")
    parser.add_argument("--timesteps", type=int, default=config.timesteps)
    runAlgo("sac", SAC, buildKwargs, parser.parse_args().timesteps)
if __name__ == "__main__": main()
