import argparse
from stable_baselines3 import A2C
import config
from pipeline import runAlgo

def buildKwargs(stockCount): return {"learning_rate": 7e-4, "n_steps": 128, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "normalize_advantage": True, "policy_kwargs": {"net_arch": [256, 256, 128]}, "verbose": 0, "seed": config.seed}
def main():
    parser = argparse.ArgumentParser(description="Train A2C with FinRL features.")
    parser.add_argument("--timesteps", type=int, default=config.timesteps)
    runAlgo("a2c", A2C, buildKwargs, parser.parse_args().timesteps)
if __name__ == "__main__": main()
