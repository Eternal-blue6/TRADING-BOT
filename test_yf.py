import yfinance as yf

sp500 = yf.Ticker("^GSPC")
print(sp500.history(period="1y"))