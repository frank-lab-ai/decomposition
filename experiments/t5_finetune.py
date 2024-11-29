import torch
import json
import os
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import nltk
nltk.download('punkt')
import numpy as np
from transformers import AutoTokenizer
import evaluate  # Bleu
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import warnings

warnings.filterwarnings("ignore")

data_files = {
    "train": "data/train.json",
    "validation": "data/validation.json",
    "test": "data/test.json"
}
max_input_length = 512
max_target_length = 512

# Load the dataset
datasets = load_dataset("json", data_files=data_files)

print("Completed Loading Datasets.")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def preprocess_data(examples):
    prefix = "decompose "
    inputs = []
    

    examples_label = examples["label"]
    examples_desc = examples["desc"]
    for i in range(len(examples_label)):
        item_label = examples_label[i]
        item_context = examples_desc[i]
        inputs.append(prefix + f"item: {item_label}.\t" + f"context: {item_context}") 

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')
    target_outputs = []
    for i in examples["part"]:
        to = ", ".join([j["plabel"] for j in i])
        target_outputs.append(to)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_outputs, max_length=max_target_length, truncation=True, padding='max_length')
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_datasets = datasets.map(preprocess_data, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "name")

batch_size = 5
model_name = "flan-t5-large"
model_dir = f"models/{model_name}"

os.makedirs(model_dir, exist_ok=True)

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="epoch",
    # eval_steps=100,
    logging_strategy="epoch",
    # logging_steps=100,
    save_strategy="epoch",
    # save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=8,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map="auto")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

metric = load_metric("rouge", trust_remote_code=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()



# get test split
test_tokenized_dataset = tokenized_datasets["test"]

test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=32)

# generate text for each batch
all_predictions = []
for i,batch in enumerate(dataloader):
  predictions = model.generate(**batch)
  all_predictions.append(predictions)

# flatten predictions
all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

# tokenize and pad titles
all_targeted_outputs = test_tokenized_dataset["labels"]

# compute metrics
predictions_labels = [all_predictions_flattened, all_targeted_outputs]
print(compute_metrics(predictions_labels))