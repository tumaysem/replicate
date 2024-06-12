
import os
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn,Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prompt import prompt
import jsonlines
from constants import SYSTEM_PROMPT,WINDOWS_LENGTH, SEQ_LEN, DECIMALS


BATCH_SIZE = 8
NUM_WORKERS = 10


class PriceDataset(Dataset):
    def __init__(self,window_len:int):
        self.window_len = window_len
        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/prices.csv'),parse_dates=['date'])
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
            
            user_prompt = prompt(statistics,current_prices)
            
            top_str = f'{{ "indice": {top_indice[0]}, "price": {round(next_max_prices[top_indice[0]].tolist(),DECIMALS)} }}'
            
            result = (
                f"[INST] <<SYS>> {SYSTEM_PROMPT} <<SYS>> "
                f"{user_prompt}[/INST]"
                f"{top_str}"
            )        
            
            writer.write({"text":result})


        
        
        
        
        
        