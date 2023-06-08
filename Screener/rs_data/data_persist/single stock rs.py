import yfinance as yf
import datetime

# Load the stock symbols from the CSV file into a list
# with open(stocks_csv_path, "r") as f:
#     symbols = [line.strip() for line in f]

# Define the number of trading days in a quarter
trading_days_per_quarter = 63


# Define a function to calculate the price percentage change for a given stock over a given number of quarters
def calculate_price_change(symbol, quarters):
    start_date = datetime.datetime.now() - datetime.timedelta(
        days=quarters * trading_days_per_quarter
    )
    end_date = datetime.datetime.now()
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    print(f"{start_date} to {end_date}")
    now_price = stock_data["Adj Close"][-1]
    print(f"now_price: {now_price}")
    past_price = stock_data["Adj Close"][0]
    print(f"past_price: {past_price}")
    price_percentage_change = (now_price / past_price) * 100
    print(f"price_percentage_change: {price_percentage_change}")
    return price_percentage_change, now_price


# Calculate the RS rating for each stock
symbol = 'XBIOW'
#rs_ratings = []
# for symbol in symbols:
try:
    if calculate_price_change(symbol,1)[1] < 10:
        print(f"{symbol} price under 10")
    else:
        c_q1 = calculate_price_change(symbol, 1)
        print(f"c_q1: {c_q1}")
        c_q2 = calculate_price_change(symbol, 2)
        print(f"c_q2: {c_q2}")
        c_q3 = calculate_price_change(symbol, 3)
        print(f"c_q3: {c_q3}")
        c_q4 = calculate_price_change(symbol, 4)
        print(f"c_q4: {c_q4}")
        rs_rating = ((0.4 * c_q1) + (0.2 * c_q2) + (0.2 * c_q3) + (0.2 * c_q4))
        print(f"rs_rating: {rs_rating}")
        #rs_ratings.append((symbol, rs_rating))
except:
    pass

# Sort the list of RS ratings by ascending order
# rs_ratings.sort(key=lambda x: x[1])

# # Get the top 30% of RS ratings
# num_top_ratings = int(len(rs_ratings) * 0.3)
# top_ratings = rs_ratings[-num_top_ratings:]

# # Write the top ratings to a CSV file
# date_string = datetime.datetime.now().strftime("%Y-%m-%d")
# filename = f"top_rs_ratings_{date_string}.csv"
# with open(filename, "w") as f:
#     f.write("Symbol,RS Rating\n")
#     for rating in top_ratings:
#         f.write(f"{rating[0]},{rating[1]}\n")
