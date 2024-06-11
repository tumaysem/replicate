import replicate

training = replicate.trainings.create(
  version="meta/llama-2-7b:73001d654114dad81ec65da3b834e2f691af1e1526453189b7bf36fb3f32d0f9",
  input={
    "train_data": "https://gist.githubusercontent.com/tumaysem/055c55b000e4c37d43ce8eb142ccc0a2/raw/d13853512fc83e8c656a3e8b6e1270dd3c398e77/output.jsonl",
    "num_train_epochs": 3
  },
  destination=f"tumay/llama2-prices"
)

print(training)