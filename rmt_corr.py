import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from marchenko_pastur import marchenko_pastur as mp

SPY_tickers = 'AAPL MSFT AMZN FB GOOGL GOOG BRK-B NVDA TSLA JPM JNJ V UNH PYPL HD PG MA DIS BAC ADBE CMCSA XOM NFLX VZ CRM INTC CSCO PFE ABT KO T PEP NKE ABBV TMO CVX MRK ACN AVGO WMT LLY WFC DHR TXN COST MCD MDT QCOM PM UPS HON ORCL LIN BMY UNP NEE C AMGN LOW INTU SBUX MS RTX BA BLK GS AMT AMAT IBM TGT CAT AMD MMM GE ISRG AXP NOW DE CVS SCHW CHTR SPGI ANTM LMT ZTS MU BKNG PLD FIS LRCX MDLZ MO SYK TMUS CCI ADP GILD TJX COP CI'
data = yf.download(
    tickers=SPY_tickers,
    period="5d",
    interval="1m"
    )
stocks = data.columns
data.dropna()
closing_data = data['Close']
closing_diff = closing_data.pct_change().dropna()
closing_diff = (closing_diff - closing_diff.mean(axis=0))
closing_diff = closing_diff/np.linalg.norm(closing_diff, axis=0)
correlation = np.matmul(closing_diff.T, closing_diff)

Q = closing_diff.shape[0]/closing_diff.shape[1]
w, v = np.linalg.eig(correlation)
v = v[w.argsort()]
w = np.sort(w)
rmt_dist = mp()
Q, var, loc, scale = rmt_dist.fit(w, f0=Q, floc=0, fscale=1)
print("variance: ", var)
sns.histplot(w, binwidth=0.2, stat='density')
x = [i for i in np.linspace(0, 10, 200)]
y = [rmt_dist.pdf(n, Q, var) for n in x]
plt.plot(x, y)
plt.show()

lambda_min = var * (1 + 1/Q - 2 * np.sqrt(1/Q))
lambda_max = var * (1 + 1/Q + 2 * np.sqrt(1/Q))
random_band = v[np.where((w < lambda_max) & (w > lambda_min))]
projection_random = np.matmul(np.matmul(random_band.T, np.linalg.inv(np.matmul(random_band, random_band.T))), random_band)
random_component = np.matmul(projection_random, correlation)
market_component = np.add(correlation, -random_component) 
cleaned_correlation = np.add(market_component, np.eye(correlation.shape[0]) * np.trace(random_component)/correlation.shape[0])

rewards = [(closing_data[col].dropna()[-1] - closing_data[col].dropna()[0]) * 100 / closing_data[col].dropna()[0] for col in closing_data]
rewards = np.asarray(rewards)

portfolio = np.matmul(np.linalg.inv(correlation), rewards)
portfolio = portfolio/np.sum(portfolio)
portfolio_c = np.matmul(np.linalg.inv(cleaned_correlation), rewards)
portfolio_c = portfolio_c/np.sum(portfolio_c)
stocks = stocks[np.argsort(portfolio_c)]
portfolio_c = np.sort(portfolio_c)

expected_reward = np.dot(rewards, portfolio)
risk = np.matmul(portfolio, np.matmul(correlation, portfolio.T))

expected_reward_c = np.dot(rewards, portfolio_c)
risk_c = np.matmul(portfolio_c, np.matmul(cleaned_correlation, portfolio_c.T))

print("Portfolio")
print("Expected return: ", expected_reward)
print("Risk: ", risk)
print("Cleaned portfolio")
print("Expected return: ", expected_reward_c)
print("Risk: ", risk_c)
print("Down:")
print(stocks[:5])
print(portfolio_c[:5])
print("Up:")
print(stocks[-5:])
print(portfolio_c[-5:])