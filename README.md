# codexfinrl

Flat FinRL-based reinforcement learning trading project for NSE equities. There are no source subfolders: every code file is directly in this folder, and the only folder is `data` for generated metrics, models, and logs.

## Files

- `config.py`: tickers, dates, capital, costs, FinRL indicators, and data path.
- `data.py`: downloads market data and applies FinRL `FeatureEngineer`.
- `environment.py`: Gymnasium trading environment.
- `metrics.py`: performance metrics and benchmark strategies.
- `pipeline.py`: shared training, validation, backtesting, and saving logic.
- `ppo.py`: PPO runner.
- `a2c.py`: A2C runner.
- `sac.py`: SAC runner.
- `td3.py`: TD3 runner.
- `data/`: output space for `<algo>_metrics.json`, `<algo>_final.zip`, best models, and logs.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Separate Algorithms

```powershell
python ppo.py --timesteps 100000
python a2c.py --timesteps 100000
python sac.py --timesteps 100000
python td3.py --timesteps 100000
```

## Results

After a run, read `data/<algo>_metrics.json`. Each metrics file compares the trained agent with Nifty 50 buy-and-hold, equal-weight buy-and-hold, and a momentum rule baseline. Models are also saved in `data` so the project stays flat.
