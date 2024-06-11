import clickhouse_connect
import os
from pandas import DataFrame,pivot_table

PRICEQUERY = '''SELECT 
block_hour AS date,token_symbol AS token,price 
FROM prices.usd_v1
WHERE block_hour > date_sub(YEAR,2,now()) 
AND token_symbol IN  ('WETH','HEX','WBTC','PEPE','SHIB','BONE','OHM','MATIC','APE','ELON','LINK','UNI','FRAX','MNT','ARB','ONDO','SAND','MASK')
ORDER BY date,token_symbol'''

def generate_prices():
    host = os.getenv('CLICKHOUSE_HOST')
    user_name = os.getenv('CLICKHOUSE_USER')
    password = os.getenv('CLICKHOUSE_PASSWORD') or ''

    if None in [host, user_name]:
        raise ValueError('CLICKHOUSE_HOST, CLICKHOUSE_USER and CLICKHOUSE_PASSWORD must be set')

    file_path = os.path.join(os.path.dirname(__file__), 'data/prices.csv')
    if os.path.exists(file_path):
        os.remove(file_path)
    client = clickhouse_connect.get_client(host=host, username=user_name, password=password, secure=True)
    raw_data:DataFrame = client.query_df(PRICEQUERY)
    result = pivot_table(raw_data,index="date",columns="token",values="price",fill_value=0.0)
    result.to_csv(file_path,sep=",",header=True)

generate_prices()