import sys
import replicate
from constants import BASE_MODEL_NAME, BASE_MODEL_VERSION, SYSTEM_PROMPT
from clickhouse import latest_prices
import torch
import re
from prompt import prompt

regex = r"\{\".*\":\s*(?P<index>\d+)\,\".*\":\s*(?P<price>\d+\.\d+)\}"

model = replicate.models.get(BASE_MODEL_NAME)

version = model.versions.get(BASE_MODEL_VERSION)

data = latest_prices()

start_date = data.index.min()
end_date = data.index.max()
tokens = data.columns.tolist()

sequence = torch.Tensor(data.values[:-1])
current = torch.Tensor(data.values[-1])

user_prompt = prompt(sequence, current)

prediction = replicate.predictions.create(
    version=version,
     input={
         "prompt": user_prompt,
        "top_k": 0,
        "top_p": 0.95,
        "max_tokens": 512,
        "temperature": 0.6,
        "system_prompt": SYSTEM_PROMPT,
        "length_penalty": 1,
        "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
        "presence_penalty": 0,
        "log_performance_metrics": False
    },
)

prediction.reload()

prediction.wait()

if prediction.error:
    print(prediction.error,file=sys.stderr)
    exit(1)

result:list[str] = prediction.output

result = ''.join(result).replace('\n','')

match = re.search(regex, result)

if match is not None:
    token = tokens[int(match.group('index'))]
    price = match.group('price')
    print(f"Token: {token} , Price: {price}")
else:
    print(f"Invalid result format {result}", file=sys.stderr)