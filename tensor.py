import yfinance as yf

data = yf.download('SPY QQQ', start = '2010-01-01')

data['Adj Close'].rebase(1).plot()

#[출처] [소개] 국내장 주가정보와 재무정보를 가져올 수 있는 파이썬 라이브러리 (FinanceDataReader, OpenDartReader)|작성자 오렌지사과