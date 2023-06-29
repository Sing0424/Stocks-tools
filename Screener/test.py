import yahooquery as yq

symbol = 'NVDA'
stock_data = yq.Ticker('NVDA')

eps_list = stock_data.income_statement("q")['DilutedEPS']
t = stock_data.earning_history

print(eps_list)
print(t)