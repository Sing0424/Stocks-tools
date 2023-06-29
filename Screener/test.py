import yahooquery as yq

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
                                                                                # GrossProfit
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