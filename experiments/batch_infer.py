import torch
import json
import sys
from tqdm import tqdm 
from template import Template
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM


if len(sys.argv) != 2:
        print("Usage: python batch_infer.py <mode>")
        sys.exit(1)
try:
    mode = sys.argv[1]
except ValueError:
    print("Argument must be strings.")
    sys.exit(1)

model_name = "t5"
x = []


with open('./data/test.json', 'r') as f:
    data = json.load(f)

all_inputs = [Template(item["label"], item["desc"].rstrip("."), mode=mode).template for item in data]

special_tokens = set(['</s>', '<unk>', '<pad>'])


model_name = "flan-t5-large"
model_dir = f"models/{model_name}"
os.makedirs(model_dir, exist_ok=True)
out_path = f"./generation/{mode}/{model_name}"
os.makedirs(out_path, exist_ok=True)

# tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto")
# model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="auto")

max_input_length = 512

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

STEP = 10

for it in tqdm(range (0, len(all_inputs), STEP)):
    
    inputs = all_inputs[it:it+STEP]
    print(inputs)
    encoding = tokenizer(inputs, padding=True, return_tensors="pt").input_ids.to(device)
    print("Started generating")
    # TODO: proofread a bunch of the model’s outputs and lower the rep penalty if necessary
    generated_ids = model.generate(encoding, max_length=1024, repetition_penalty=2.0)
    print("Done")
    result = {}

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for i in range(len(inputs)):
        item_result = dict()
        item_result["label"] = data[it+i]["label"]
        if len(output[i].split(", ")) == 1:
            item_result["output"] = output[i]
        else:
            item_result["output"] = output[i].split(", ")
        item_result["reference"] = [p["plabel"] for p in data[it+i]["part"]]
        result[data[it+i]["qid"]] = item_result

    print(os.getcwd())
    fn = f"test_{it//STEP+1}.json"
    with open(os.path.join(out_path,fn), 'w') as f:
        json.dump(result, f, indent = 2)

