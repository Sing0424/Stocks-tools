import yahooquery as yq
import pandas as pd

symbol = 'SMCI'

yq_stock_data = yq.Ticker(symbol)
inc_stat = yq_stock_data.income_statement('q', trailing=False)
inc_list = pd.DataFrame(inc_stat['NetIncome']).dropna()
lenth_inc_list = len(inc_list)

first_qtr_inc = inc_list.iloc[lenth_inc_list-4,0]

second_qtr_inc = inc_list.iloc[lenth_inc_list-3,0]

third_qtr_inc = inc_list.iloc[lenth_inc_list-2,0]

current_qtr_inc = inc_list.iloc[lenth_inc_list-1,0]

print(inc_list)
print('-------------------------------------------------------------------------------')
print(first_qtr_inc)
print(second_qtr_inc)
print(third_qtr_inc)
print(current_qtr_inc)