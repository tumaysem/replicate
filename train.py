import replicate
from constants import BASE_MODEL_NAME, BASE_MODEL_VERSION, MODEL_NAME


training = replicate.trainings.create(
  version=f"{BASE_MODEL_NAME}:{BASE_MODEL_VERSION}",
  input={
    "train_data": "https://raw.githubusercontent.com/tumaysem/replicate/main/output.jsonl",
    "num_train_epochs": 3
  },
  destination=MODEL_NAME
)

print(training)