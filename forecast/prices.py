import clickhouse_connect
import os
from pandas import DataFrame,pivot_table
from forecast.constants import TOKENS, YEARS, SEQ_LEN
from torch import Tensor

TOKEN_STR = ','.join([f'\'{token}\' ' for token in TOKENS])

PRICEQUERY = f'''SELECT 
block_hour AS date,token_symbol AS token,price 
FROM prices.usd_v1
WHERE block_hour > date_sub(YEAR,{YEARS},now()) 
AND token_symbol IN  ({TOKEN_STR})
ORDER BY date,token_symbol'''

LATEST_PRICES = f'''SELECT 
block_hour AS date,token_symbol AS token,price 
FROM prices.usd_v1
WHERE block_hour > date_sub(HOUR,{SEQ_LEN + 2},now()) 
AND token_symbol IN  ({TOKEN_STR})
ORDER BY date,token_symbol'''

def generate_prices():
    host = os.getenv('CLICKHOUSE_HOST')
    user_name = os.getenv('CLICKHOUSE_USER')
    password = os.getenv('CLICKHOUSE_PASSWORD') or ''

    if None in [host, user_name]:
        raise ValueError('CLICKHOUSE_HOST, CLICKHOUSE_USER and CLICKHOUSE_PASSWORD must be set')
    file_path = os.path.join(os.path.dirname(__file__), '../data/prices.csv')
    if os.path.exists(file_path):
        os.remove(file_path)

    print(f'Connecting to Clickhouse at {host} as {user_name}')
    client = clickhouse_connect.get_client(host=host, username=user_name, password=password, secure=True)
    print(f'Querying recent {YEARS} year prices from Clickhouse')
    raw_data:DataFrame = client.query_df(PRICEQUERY)
    print(f'Pivotting data (rows:{raw_data.shape[0]}) to with tokens')
    result = pivot_table(raw_data,index="date",columns="token",values="price",fill_value=0.0)
    print(f'Saving data to {file_path}')
    result.to_csv(file_path,sep=",",header=True)
    print(f'Done')

def latest_prices() -> DataFrame:
    host = os.getenv('CLICKHOUSE_HOST')
    user_name = os.getenv('CLICKHOUSE_USER')
    password = os.getenv('CLICKHOUSE_PASSWORD') or ''
    
    if None in [host, user_name]:
        raise ValueError('CLICKHOUSE_HOST, CLICKHOUSE_USER and CLICKHOUSE_PASSWORD must be set')
    
    print(f'Connecting to Clickhouse at {host} as {user_name}')
    client = clickhouse_connect.get_client(host=host, username=user_name, password=password, secure=True)
    print(f'Querying latest prices from Clickhouse')
    raw_data:DataFrame = client.query_df(LATEST_PRICES)
    result = pivot_table(raw_data,index="date",columns="token",values="price",fill_value=0.0)
    return result
