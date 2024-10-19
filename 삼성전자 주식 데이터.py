import yfinance as yf

# 예시: 삼성전자 주식 데이터를 불러오기
stock_data = yf.download('005930.KS', start='2020-01-01', end='2023-01-01')
stock_data.to_csv('samsung_stock.csv')  # CSV로 저장
