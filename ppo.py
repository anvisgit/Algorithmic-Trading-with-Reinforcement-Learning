import argparse
from stable_baselines3 import PPO
import config
from pipeline import runAlgo

def buildKwargs(stockCount): return {"learning_rate": 2.5e-4, "n_steps": 4096, "batch_size": 128, "n_epochs": 20, "gamma": 0.995, "gae_lambda": 0.95, "clip_range": 0.2, "vf_coef": 0.5, "ent_coef": 0.01, "max_grad_norm": 0.5, "target_kl": 0.03, "policy_kwargs": {"net_arch": [256, 256, 128]}, "verbose": 0, "seed": config.seed}
def main():
    parser = argparse.ArgumentParser(description="Train PPO with FinRL features.")
    parser.add_argument("--timesteps", type=int, default=config.timesteps)
    runAlgo("ppo", PPO, buildKwargs, parser.parse_args().timesteps)
if __name__ == "__main__": main()
