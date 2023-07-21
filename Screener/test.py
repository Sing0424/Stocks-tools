import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf

# List of stock symbols
symbols = ['AAPL']

for symbol in symbols:
    
    # Get stock data 
    data = yf.Ticker(symbol)
    df = data.history(period="1d", start="2022-1-1", end="2023-7-21")
    
    # Plotting
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 6) ,sharex = True, height_ratios = [4, 1])
    # fig.suptitle('Stock price and volume of {}'.format(symbol))

    # Price chart
    # ax1.plot(df.index, df['Close'], color='r', label='Close price', mav=(10,20,30,50,150,200))     
    # ax1.set_ylabel('Price')
    
    # # Volume chart      
    # ax2.bar(df.index, df['Volume'], color='b', label='Volume')       
    # ax2.set_ylabel('Volume')
    
    # plt.legend(loc='best')    
    # plt.show()  

    mc = mpf.make_marketcolors(up='g',down='r',inherit=True)
    s  = mpf.make_mpf_style(base_mpf_style='yahoo',marketcolors=mc)
    #針對線圖的外觀微調，將上漲設定為紅色，下跌設定為綠色，符合台股表示習慣
    #接著把自訂的marketcolors放到自訂的style中，而這個改動是基於預設的yahoo外觀

    kwargs = dict(type='candle', mav=(5,20,60), volume=True, figratio=(10,8), figscale=0.75, title='NVDA', style=s) 
    #設定可變參數kwargs，並在變數中填上繪圖時會用到的設定值

    mpf.plot(df, **kwargs)
    #選擇df資料表為資料來源，帶入kwargs參數，畫出目標股票的走勢圖