import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I got alot of objectives in my life",
    "CSGO 2 is filled with cheaters!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)