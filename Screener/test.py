import yahooquery as yq
import pandas as pd

<<<<<<< HEAD
symbol = 'EC'

eps = yq.Ticker(symbol).income_statement()['DilutedEPS']

print(eps)

eps_first_y = eps[0]
eps_second_y = eps[1]
eps_third_y = eps[2]
eps_current_y = eps[3]

print(f'first year eps: {eps_first_y}')
print(f'second year eps: {eps_second_y}')
print(f'third year eps: {eps_third_y}')
print(f'current year eps: {eps_current_y}')

=======
symbol = 'SMCI'
stock_data = yq.Ticker(symbol)

inc_stat = stock_data.income_statement('q')
inc_list = pd.DataFrame(inc_stat['NetIncome']).dropna()
lenth_inc_list = len(inc_list)
first_qtr_inc = inc_list.iloc[lenth_inc_list-5,0]
second_qtr_inc = inc_list.iloc[lenth_inc_list-4,0]
current_qtr_inc = inc_list.iloc[lenth_inc_list-2,0]

# net_inc_list = inc_stat['NetIncome']
# first_yr_net_inc = net_inc_list[0]
# second_yr_net_inc = net_inc_list[1]
# third_yr_net_inc = net_inc_list[2]
# fourth_yr_net_inc = net_inc_list[3]

# rev_list = inc_stat['TotalRevenue']
# first_yr_rev = rev_list[0]
# second_yr_rev = rev_list[1]
# third_yr_rev = rev_list[2]
# fourth_yr_rev = rev_list[3]


# Net Profit Margin = Net income/Total Revenue * 100.
>>>>>>> b2218a48e853e4aee2ce95b7d9215565788d9efc

# asOfDate
# periodType
# currencyCode
# BasicAverageShares
# BasicEPS
# CostOfRevenue
# DilutedAverageShares
<<<<<<< HEAD
                                                                                # DilutedEPS
# DilutedNIAvailtoComStockholders
# EBIT
# EBITDA
                                                                                # GrossProfit
=======
                                                                    # DilutedEPS
# DilutedNIAvailtoComStockholders
# EBIT
# EBITDA
>>>>>>> b2218a48e853e4aee2ce95b7d9215565788d9efc
# InterestExpense
# InterestExpenseNonOperating
# InterestIncome
# InterestIncomeNonOperating
<<<<<<< HEAD
                                                                                # NetIncome
=======
                                                                    # NetIncome
>>>>>>> b2218a48e853e4aee2ce95b7d9215565788d9efc
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
<<<<<<< HEAD
# TotalRevenue
=======
                                                                # TotalRevenue
>>>>>>> b2218a48e853e4aee2ce95b7d9215565788d9efc
# TotalUnusualItems
# TotalUnusualItemsExcludingGoodwill