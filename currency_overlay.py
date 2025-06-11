import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cp
from scipy.optimize import minimize
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Step 1: Define ETF and Currency Information
ASSET_INFO = [
    {
        "country": "Japan",
        "currency": "JPY",
        "fx_ticker": "JPYUSD=X",
        "fx_type": "inverse",
        "weight": 0.20,
        "etf_ticker": "EWJ",
    },
    {
        "country": "Canada",
        "currency": "CAD",
        "fx_ticker": "CADUSD=X",
        "fx_type": "inverse",
        "weight": 0.05,
        "etf_ticker": "EWC",
    },
    {
        "country": "Australia",
        "currency": "AUD",
        "fx_ticker": "AUDUSD=X",
        "fx_type": "direct",
        "weight": 0.05,
        "etf_ticker": "EWA",
    },
    {
        "country": "Europe",
        "currency": "EUR",
        "fx_ticker": "EURUSD=X",
        "fx_type": "direct",
        "weight": 0.7,
        "etf_ticker": "VGK",
    },
]

# Calculate benchmark weights
benchmark_weights = np.array([asset["weight"] for asset in ASSET_INFO])
benchmark_weights = benchmark_weights / benchmark_weights.sum()

# Plot portfolio weights
asset_names = [asset["country"] for asset in ASSET_INFO]
df_weight = pd.DataFrame(
    {"Ticker": asset_names, "Weight": benchmark_weights}
).sort_values(by="Weight", ascending=True)
plt.figure(figsize=(12, 6))
plt.barh(df_weight["Ticker"], df_weight["Weight"] * 100)
plt.xlabel("Weight")
plt.title("Portfolio Weights")
plt.grid(axis="x")
plt.show()

# Step 2: Download Historical Data
start_date = "2020-01-01"
end_date = "2025-05-31"

raw_data = {}
tickers = [asset["etf_ticker"] for asset in ASSET_INFO] + [
    asset["fx_ticker"] for asset in ASSET_INFO
]
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    raw_data[ticker] = df

# Step 3: Process Data
# Use 'Close' prices for simplicity

prices = pd.DataFrame(
    {ticker: raw_data[ticker]["Close"].squeeze() for ticker in tickers}
).dropna()

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(prices["EWJ"], color="red", label="EWJ")
ax1.set_ylabel("EWJ", color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax2 = ax1.twinx()
ax2.plot(prices["JPYUSD=X"], color="blue", label="JPYUSD=X")
ax2.set_ylabel("JPY/USD FX Rate", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
plt.title("EWJ vs JPY/USD")
fig.tight_layout()
plt.show()

# Process returns data

returns_df = prices.pct_change().dropna()
etf_tickers = [asset["etf_ticker"] for asset in ASSET_INFO]
currency_tickers = [asset["fx_ticker"] for asset in ASSET_INFO]

etf_returns = returns_df[etf_tickers]
fx_returns = returns_df[currency_tickers]

plt.figure(figsize=(12, 10))
for ticker in etf_returns.columns:
    plt.plot((1 + etf_returns[ticker]).cumprod(), label=ticker)
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid()
plt.show()

# Step 4: Annualize Returns

equity_return_unhedged = etf_returns @ benchmark_weights
er_unhedged_equity_return = etf_returns.mean()
unhedged_annual_return = er_unhedged_equity_return @ benchmark_weights * 252

mu_hat_annual = fx_returns.mean() * 252
cov_matrix_fx_annual = fx_returns.cov().to_numpy() * 252
cov_r_unhedged_fx_annual = (
    fx_returns.apply(lambda col: col.cov(equity_return_unhedged)).to_numpy() * 252
)
var_r_unhedged_annual = equity_return_unhedged.var() * 252

# Correlations and volatilities
corr_matrix_fx = fx_returns.corr()
vol_fx_annual = fx_returns.std() * np.sqrt(252)
vol_equity_annual = equity_return_unhedged.std() * np.sqrt(252)
cov_matrix_fx_annual = corr_matrix_fx.to_numpy() * np.outer(
    vol_fx_annual, vol_fx_annual
)

corr_equity_fx = fx_returns.corrwith(equity_return_unhedged)
cov_r_unhedged_fx_annual = corr_equity_fx * vol_fx_annual * vol_equity_annual
var_r_unhedged_annual = vol_equity_annual**2

# Optimization parameters
weights = benchmark_weights
N_CURRENCIES = len(currency_tickers)
TOLERANCE = 0.001
h = cp.Variable(N_CURRENCIES)

net_exposure_benchmark = (1 - 0.5) * weights
benchmark_fx_te = np.sqrt(
    net_exposure_benchmark.T @ cov_matrix_fx_annual @ net_exposure_benchmark
)
benchmark_fx_beta = (
    cov_r_unhedged_fx_annual @ net_exposure_benchmark
) / var_r_unhedged_annual

# Objective function


def objective(h_vec):
    overlay_return = -np.dot(h_vec, mu_hat_annual * weights)
    return -(unhedged_annual_return + overlay_return)


# Portfolio volatility


def portfolio_volatility(h_vec):
    x = (1 - h_vec) * weights
    var_fx = x.T @ cov_matrix_fx_annual @ x
    cov_r_fx = cov_r_unhedged_fx_annual @ x
    var_p = var_r_unhedged_annual + var_fx + 2 * cov_r_fx
    return np.sqrt(var_p)


# Tracking error


def fx_tracking_error(h_vec):
    x = (1 - h_vec) * weights
    return np.sqrt(x.T @ cov_matrix_fx_annual @ x)


# Equity beta


def fx_equity_beta(h_vec):
    x = (1 - h_vec) * weights
    cov_fx_equity = cov_r_unhedged_fx_annual @ x
    return cov_fx_equity / var_r_unhedged_annual


# Constraints


def constraints_with_vol(target_vol):
    return [
        {
            "type": "ineq",
            "fun": lambda h_vec: target_vol + TOLERANCE - portfolio_volatility(h_vec),
        },
        {
            "type": "ineq",
            "fun": lambda h_vec: benchmark_fx_te + TOLERANCE - fx_tracking_error(h_vec),
        },
        {
            "type": "ineq",
            "fun": lambda h_vec: benchmark_fx_beta + TOLERANCE - fx_equity_beta(h_vec),
        },
    ]


bounds = [(1 - 0.2 / weights[i], 1 + 0.2 / weights[i]) for i in range(N_CURRENCIES)]

# Efficient frontier

target_volatilities = np.linspace(0.05, 0.35, 20)
volatilities = []
returns = []
sharpe_ratios = []
hedge_ratios = []
risk_free_rate = 0.02

for target_vol in target_volatilities:
    cons = constraints_with_vol(target_vol)
    result = minimize(
        objective,
        np.zeros(N_CURRENCIES),
        bounds=bounds,
        method="SLSQP",
        constraints=cons,
        options={"maxiter": 1000},
    )
    if result.success:
        h_opt = result.x
        vol = portfolio_volatility(h_opt)
        ret = -objective(h_opt)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        volatilities.append(vol)
        returns.append(ret)
        sharpe_ratios.append(sharpe)
        hedge_ratios.append(h_opt.copy())

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(volatilities, returns, "b-", label="Efficient Frontier")
if volatilities:
    plt.scatter(volatilities[0], returns[0], c="r", label="Min Volatility")
    plt.scatter(volatilities[-1], returns[-1], c="g", label="Max Return")
plt.xlabel("Portfolio Volatility (Annualized)")
plt.ylabel("Expected Return (Annualized)")
plt.title("Efficient Frontier for Currency Hedging Strategy")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(volatilities, sharpe_ratios, "m-", label="Sharpe Ratio")
plt.xlabel("Portfolio Volatility (Annualized)")
plt.ylabel("Sharpe Ratio")
plt.title("Sharpe Ratio Across Efficient Frontier")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

hedge_ratios = np.array(hedge_ratios)
if len(hedge_ratios) != len(volatilities):
    min_length = min(len(hedge_ratios), len(volatilities))
    hedge_ratios = hedge_ratios[:min_length]
    volatilities = volatilities[:min_length]
    returns = returns[:min_length]
    sharpe_ratios = sharpe_ratios[:min_length]

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(volatilities, returns, "b-", label="Efficient Frontier")
if volatilities:
    plt.scatter(volatilities[0], returns[0], c="r", label="Min Volatility")
    plt.scatter(volatilities[-1], returns[-1], c="g", label="Max Return")
plt.xlabel("Portfolio Volatility (Annualized)")
plt.ylabel("Expected Return (Annualized)")
plt.title("Efficient Frontier for Currency Hedging Strategy")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
bar_width = 0.1
index = np.arange(len(volatilities))
if len(volatilities) > 0:
    for i in range(N_CURRENCIES):
        plt.bar(
            index + i * bar_width,
            hedge_ratios[:, i],
            bar_width,
            label=currency_tickers[i],
        )
    plt.xlabel("Target Volatility Index")
    plt.ylabel("Hedge Ratio")
    plt.title("Optimal Hedge Ratios Across Efficient Frontier")
    plt.xticks(
        index + bar_width * (N_CURRENCIES - 1) / 2, [f"{v:.2f}" for v in volatilities]
    )
    plt.legend()
    plt.grid(True)
else:
    print("No valid optimization results to plot.")

plt.tight_layout()
plt.show()
