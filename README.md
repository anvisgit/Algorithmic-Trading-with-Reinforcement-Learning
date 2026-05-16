# Algorithmic Trading with Reinforcement Learning

This repository implements an RL-based trading system for Indian equities using a PPO agent. It is inspired by the FinRL research framework with custom handling for Indian market costs and NSE data.
## In a nutshell: 
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Market:** NSE Indian stocks (sample tickers: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK)
- **Training period:** 2018-01-01 to 2022-12-31
- **Test period:** 2023-01-01 to 2024-12-31
- **Initial capital:** ₹1,000,000
- **Transaction costs:** STT, brokerage, GST, slippage

## Workflow:

1. Download NSE market price data with `yfinance`
2. Compute technical indicators and features
3. Build a custom trading environment for multi-stock portfolio control
4. Train a PPO model with a validation callback
5. Backtest the trained model on the test set
6. Compare performance against three baselines:
   - Nifty 50 buy-and-hold
   - Equal-weight buy-and-hold
   - Momentum rule strategy

## Evaluation Analysis

The current evaluation shows that the PPO agent can deliver a much higher final portfolio value compared to the Nifty baseline. However, the trading strategy is also significantly riskier.

### Key findings

- **Nifty 50 (B&H)**
  - Total return: ~30%
  - Annual return: ~14.4%
  - Max drawdown: ~-10.9%
  - Calmar ratio: ~1.32
  - Sharpe ratio: ~0.65

- **PPO agent**
  - Final portfolio can be substantially higher than benchmark
  - Drawdown is extreme (~-78%)
  - Calmar ratio is below the benchmark (~0.78)
  - Win rate is lower than the benchmark

### Interpretation

- The PPO agent currently appears to be a **high-return, high-risk** strategy.
- The Nifty benchmark is **more stable** and offers better risk-adjusted performance.



