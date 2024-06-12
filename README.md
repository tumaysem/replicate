# Time Series Forecasting by fine-tuning a LLM model in replicate
Replicate provides a inference and fine-tuning LLM services. We try to use the 
[TimeLLM](https://arxiv.org/abs/2310.01728) paper base as a to fine-tune llama-2-7b
for forcasting best gaining value token on ethereum network.


## Installation 
Use `poetry` install dependencies:

```console
git clone https://github.com/tumaysem/replicate.git
cd replicate
poetry install
```

## Usage
We used our (agentc.xyz)[https://agentc.xyz] prices clickhouse database to
generate fine tune prompts and push it to replicate for fine-tune.

Create your `.env` file ,see example.

### Prices data

Collect hourly price data fro the last `constants.YEARS`.

```console
poetry run prices
```

### Train prompts
Converts prices data into propmts file for fine tuning. Replicate uses publicly
accessible files for fine-tuning.

```console
poetry run prices
```

### Initiate a fine-tune training on replicate
Uses the prompts to fine tune llama-2-7b-chat for forecasting

```console
poetry run train
```

### Forecast best value gaining token
Uses fine-tuned LLM to forecast most value gaining token based on statistic data
of previous token prices.

```console
poetry run forecast
```




