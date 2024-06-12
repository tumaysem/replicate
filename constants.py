BASE_MODEL_NAME = 'meta/llama-2-7b-chat'
BASE_MODEL_VERSION = '13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0'
MODEL_NAME = 'tumaysem/llama2-prices'

SYSTEM_PROMPT = "You are a code generator. Always output your answer in JSON. No pre-amble. Only token indice and price."
TOP_K = 5
DECIMALS = 3
TOKENS = ['WETH','HEX','WBTC','PEPE','SHIB','BONE','OHM','MATIC','APE','ELON','LINK','UNI','FRAX','MNT','ARB','ONDO','SAND','MASK']
YEARS = 2
WINDOWS_LENGTH = 24
PRED_LEN = 5
SEQ_LEN = WINDOWS_LENGTH - PRED_LEN - 1