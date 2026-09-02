# JLST BTC Reversal-Momentum Dashboard

Live analysis of BTC using the reversal–momentum composite model from:

> **Jegadeesh, Luo, Subrahmanyam & Titman (2025)**  
> *"Short-Term Reversals and Longer-Term Momentum around the World: Theory and Evidence"*  
> *Review of Financial Studies*

## What This Does

The JLST paper establishes that stock returns exhibit:
- **Short-term reversals** (ρ₁ < 0): past 1-month returns negatively predict future returns
- **A transition zone** (ρ₂ ≈ 0): 2-month lag is insignificant
- **Longer-term momentum** (ρ₃…ρ₁₂ > 0): 3–12 month lags are positively predictive

This dashboard adapts the model for **BTC**, computing a composite reversal-momentum score with:
- Log returns (for high-volatility assets)
- 24/7 calendar (30-day months, 365-day years)
- Dynamic noise-based weight adjustment (Prediction c)
- Information shock detection replacing earnings announcements (Prediction a)
- Regime confidence monitoring via corr(Rev, Mom) (Prediction b)
- BTC-specific modules: halving cycle, cascade detection

## Live Site

**[→ View Dashboard](https://hhysteric.github.io/btc-jlst-momentum/)**

## Data Pipeline

Data is fetched from **Binance** (BTCUSDT) and updated daily via GitHub Actions.

Supported timeframes: **Daily** / **Weekly** / **Monthly**

### Run Locally

```bash
pip install -r requirements.txt
python scripts/update_data.py
```

Then open `index.html` in a browser.

## Disclaimer

For quantitative research and educational purposes only. Not investment advice. The paper's returns come from cross-sectional decile hedge portfolios; this is a single-asset time-series approximation.
