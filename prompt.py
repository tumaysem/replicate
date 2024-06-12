import torch
from torch import Tensor
from constants import TOP_K, DECIMALS, PRED_LEN


def calcute_lags(values:Tensor):
    q_fft = torch.fft.rfft(values.contiguous(), dim=-1)
    k_fft = torch.fft.rfft(values.contiguous(), dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value,TOP_K, dim=-1)
    return lags

def prompt(sequence:Tensor,current:Tensor) -> str:
        seq_len = sequence.shape[0]
        token_len = sequence.shape[1]
        sequence.transpose_(0,1)
        mins = sequence.min(dim=1).values
        maxs = sequence.max(dim=1).values
        medians = sequence.median(dim=1).values
        lags = calcute_lags(sequence)
        trends = sequence.diff(dim=1).sum(dim=1)
            
        mins_values_str = ', '.join([str(round(min,DECIMALS)) for min in mins.tolist()])
        maxs_values_str = ', '.join([str(round(max,DECIMALS)) for max in maxs.tolist()])
        median_values_str = ', '.join([str(round(median,DECIMALS)) for median in medians.tolist()])
        lags_values_str = ', '.join([str(round(lag,DECIMALS)) for lag in lags.tolist()])
        current_prices_str = ', '.join([str(round(price,DECIMALS)) for price in current.tolist()])
        trends_values_tr = ', '.join([ "up" if trend > 0 else "down" for trend in trends.tolist()])

        prompt = (
            f"Forecast most profitable single token for the next {PRED_LEN} hours from information of {token_len} tokens for previous {seq_len} hours; "
            f"min values are [{mins_values_str}], "
            f"max values are [{maxs_values_str}], "
            f"median values are [{median_values_str}], "
            f"the trend of inputs are [{trends_values_tr}], "
            f"top {TOP_K} lag token indexes are [{lags_values_str}] "
            f"and  current prices are [{current_prices_str}]."
        )  
        
        return prompt