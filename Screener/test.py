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