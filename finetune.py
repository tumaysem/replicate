
import os
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn,Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import jsonlines

BATCH_SIZE = 8
WINDOWS_LENGTH = 24
PRED_LEN = 5
SEQ_LEN = WINDOWS_LENGTH - PRED_LEN - 1
NUM_WORKERS = 10
TOP_K = 5
DECIMALS = 3

def calcute_lags(values:Tensor):
    q_fft = torch.fft.rfft(values.contiguous(), dim=-1)
    k_fft = torch.fft.rfft(values.contiguous(), dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value,TOP_K, dim=-1)
    return lags


class PriceDataset(Dataset):
    def __init__(self,window_len:int):
        self.window_len = window_len
        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/prices.csv'),parse_dates=['date'])
        self.tokens = self.data.columns[1:]
        self.tot_len = len(self.data) // (self.window_len) 

    def __getitem__(self, index) -> Tensor:
        start_index = index * self.window_len;
        end_index = start_index + self.window_len;
        return Tensor(self.data.iloc[start_index:end_index,1:].values)

    def __len__(self):
        return self.tot_len

data = PriceDataset(WINDOWS_LENGTH)

loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)  

with jsonlines.open('output.jsonl', 'w') as writer:
    for i, batch in tqdm(enumerate(loader)):
        batch:Tensor = batch
        for j in range(0,batch.size()[0]):
            statistics = batch[j,:SEQ_LEN]   
            current_prices = batch[j,SEQ_LEN]
            next_max_prices = batch[j,SEQ_LEN+1:].transpose(0,1).max(dim=1).values
            _,top_indice = next_max_prices.sub(current_prices).topk(1)
            
            statistics.transpose_(0,1)
            mins = statistics.min(dim=1).values
            maxs = statistics.max(dim=1).values
            medians = statistics.median(dim=1).values
            lags = calcute_lags(statistics)
            trends = statistics.diff(dim=1).sum(dim=1)
                
            mins_values_str = ', '.join([str(round(min,DECIMALS)) for min in mins.tolist()])
            maxs_values_str = ', '.join([str(round(max,DECIMALS)) for max in maxs.tolist()])
            median_values_str = ', '.join([str(round(median,DECIMALS)) for median in medians.tolist()])
            lags_values_str = ', '.join([str(round(lag,DECIMALS)) for lag in lags.tolist()])
            current_prices_str = ', '.join([str(round(price,DECIMALS)) for price in current_prices.tolist()])
            trends_values_tr = ', '.join([ "up" if trend > 0 else "down" for trend in trends.tolist()])
            
            top_str = f'{{ "indice": {top_indice[0]}, "price": {round(next_max_prices[top_indice[0]].tolist(),DECIMALS)} }}'
            
            prompt = (
                f"[INST] <<SYS>> You are a code generator. Always output your answer in JSON. No pre-amble. Only token indice and price.<<SYS>>"
                f"Forecast most profitable token for the next 5 hours from current prices and 16 hour statistics (min, max, lag, mean and trend) of 17 tokens ;"
                f"min values are [{mins_values_str}], "
                f"max values are [{maxs_values_str}], "
                f"median values are [{median_values_str}], "
                f"the trend of inputs are [{trends_values_tr}], "
                f"top {TOP_K} lag token indexes are [{lags_values_str}] "
                f"and  current prices are [{current_prices_str}]."
                "[/INST]"
                f"{top_str}"
            )        
            
            writer.write({"text":prompt})


        
        
        
        
        
        