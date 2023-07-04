import yahooquery as yq
import pandas as pd

symbol = 'SMCI'
stock_data = yq.Ticker(symbol)

inc_stat = stock_data.income_statement('q')

inc_list = pd.DataFrame(inc_stat['NetIncome']).dropna()
lenth_inc_list = len(inc_list)
first_qtr_inc = inc_list.iloc[lenth_inc_list-5,0]
second_qtr_inc = inc_list.iloc[lenth_inc_list-4,0]
current_qtr_inc = inc_list.iloc[lenth_inc_list-2,0]

rev_list = pd.DataFrame(inc_stat['TotalRevenue']).dropna()
lenth_rev_list = len(rev_list)
first_qtr_rev = 0
second_qtr_rev = rev_list.iloc[lenth_rev_list-4,0]
current_qtr_rev = rev_list.iloc[lenth_rev_list-2,0]

if first_qtr_rev != 0 and second_qtr_rev !=0 and current_qtr_rev !=0:
    first_qtr_profit_margin = first_qtr_inc/first_qtr_rev * 100
    second_qtr_profit_margin = second_qtr_inc/second_qtr_rev * 100
    current_qtr_profit_margin = current_qtr_inc/current_qtr_rev * 100
else:
    first_qtr_profit_margin = 0
    second_qtr_profit_margin = 0
    current_qtr_profit_margin = 0

print(first_qtr_profit_margin)
print(second_qtr_profit_margin)
print(current_qtr_profit_margin)

# Net Profit Margin = Net income/Total Revenue * 100.

# asOfDate
# periodType
# currencyCode
# BasicAverageShares
# BasicEPS
# CostOfRevenue
# DilutedAverageShares
                                                                    # DilutedEPS
# DilutedNIAvailtoComStockholders
# EBIT
# EBITDA
# InterestExpense
# InterestExpenseNonOperating
# InterestIncome
# InterestIncomeNonOperating
                                                                    # NetIncome
# NetIncomeCommonStockholders
# NetIncomeContinuousOperations
# NetIncomeFromContinuingAndDiscontinuedOperation
# NetIncomeFromContinuingOperationNetMinorityInterest
# NetIncomeIncludingNoncontrollingInterests
# NetInterestIncome
# NetNonOperatingInterestIncomeExpense
# NormalizedEBITDA
# NormalizedIncome
# OperatingExpense
# OperatingIncome
# OperatingRevenue
# OtherIncomeExpense
# OtherNonOperatingIncomeExpenses
# PretaxIncome
# ReconciledCostOfRevenue
# ReconciledDepreciation
# ResearchAndDevelopment
# RestructuringAndMergernAcquisition
# SellingGeneralAndAdministration
# SpecialIncomeCharges
# TaxEffectOfUnusualItems
# TaxProvision
# TaxRateForCalcs
# TotalExpenses
# TotalOperatingIncomeAsReported
                                                                # TotalRevenue
# TotalUnusualItems
# TotalUnusualItemsExcludingGoodwill