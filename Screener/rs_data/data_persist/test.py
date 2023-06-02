import json

def method6(): 
  with open('test_tickers_info.json', 'r') as exist_json:
        json_str = exist_json.read()
        exist_data = json.loads(json_str)
  print(exist_data)

def method5():
    with open('test_tickers_info.json', 'r') as exist_json:
        json_str = exist_json.read()
        exist_data = json.loads(json_str)

    from io import StringIO
    file_str = StringIO()
    for key in exist_data:
      file_str.write(key + '\n')

    print(file_str.getvalue())

method5()