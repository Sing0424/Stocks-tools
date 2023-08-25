rs_rating_list = [0,1,2,3,4,5,6,7,8,9,10]
top_rating =0.2

rs_rating_list.sort(key=lambda x: x, reverse=True)

num_top_ratings = int(len(rs_rating_list) * top_rating)
top_ratings = rs_rating_list[:num_top_ratings]

print(top_ratings)