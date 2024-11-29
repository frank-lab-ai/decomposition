import os
import sys
from template import Template
from groq import Groq
from tqdm import tqdm
import json
import time

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
)

with open('./data/test.json', 'r') as f:
    data = json.load(f)

if len(sys.argv) != 2:
        print("Usage: python llama_infer.py <mode>")
        sys.exit(1)
try:
    mode = sys.argv[1]
except ValueError:
    print("Argument must be strings.")
    sys.exit(1)

model_name = "llama-3.1"
    
def make_few_shot_prefix(n=7):
        prefix = ""
        with open("data/in_context_held_out.json","r") as f:
            data = json.load(f)
        for i in data[:n]:
            item = i["label"]
            context = i["desc"]
            temp = i["part"]
            temp = [i["plabel"] for i in temp]
            parts = ", ".join(temp)
            prefix += f"Item: {item}\nContext: {context}\nParts: {parts}\n\n"
        return prefix 

def make_few_shot(item_batch):
    
    prefix = make_few_shot_prefix(5)

    prompt = ""
    for x in item_batch:
        label, context = x["label"], x["desc"]
        prompt += f"Item: {label}\nContext: {context}\nParts: \n"

    example = ['For an item and its context, you list all of the possible parts of that item in a single line and in this format: "[part], [part], ...". Here are some examples:\n\n',
        prefix, "\nNow you try:\n", prompt]
    example = "".join(example)
    return example

def write_template(item_batch, mode):
    if mode == "few-shot-context":
        content = make_few_shot(item_batch)

    else:    
        content = "List all possible parts of each item listed below. Return a JSON object where each key is the item provided as string (with double quote) and the value is a python list (must in square bracket) of its part. The items in list should be formatted as string with double quote as well." + \
                "Do not describe the parts and do not number the parts. \n" + "Please only produce JSON that can be parsed into a json object without an error."
        for item in item_batch:
            content += (item["label"] + "\n")

        # Add definition as prefix
        if mode == "zero-shot-context":
            context = ""
            for item in item_batch:
                label, desc = item["label"], item["desc"]
                context += f"{label} is the {desc}.\n"
            content = context + content
    

    message = [
            {
                "role": "user",
                "content": content
            },
        ]
    return message

BATCH_SIZE = 10

start = 0

out_path = f"./generation/{mode}/{model_name}"
os.makedirs(out_path, exist_ok=True)

data = data
for it in tqdm(range(start*BATCH_SIZE, len(data), BATCH_SIZE)):
     inputs = data[it: it+BATCH_SIZE]
     
     message = write_template(inputs, mode)
     retry = True
     while retry:
        completion = client.chat.completions.create(
            messages=message,
            model="llama-3.1-70b-versatile",
            temperature=1,
            max_tokens=1000,
            )
        if mode == "few-shot-context":
            results = completion.choices[0].message.content
        else:
            results = json.loads(completion.choices[0].message.content)

        parts = []
        items_with_parts = []
        for line in[s for s in results.splitlines() if s]:
            if it < len(data)-1:
                if "Parts:" not in line:
                    continue
            
                #  items_with_parts.append(line[:line.find(":")])
            line = line[line.find(":")+1:]
            res = line.split(",")
            res = [x.strip() for x in res if x.strip()]
            if len(res) > 0:
                parts.append(res)
        parts = [p for p in parts if p]
        
        if it < len(data) - 1:
            if len(parts) == len(inputs):
                retry = False
        else:
            print(results)
            print(parts)
            print("Trying again")

        out_batch = []
        for i, ps in enumerate(parts):
            item_result = dict()
            item_result["qid"] = data[it+i]["qid"]
            item_result["label"] = data[it+i]["label"]
            
            item_result["output"] = ps
            item_result["reference"] = [p["plabel"] for p in data[it+i]["part"]]

            out_batch.append(item_result)
            
        fn = f'test_{it//BATCH_SIZE+1}.json'
        with open(os.path.join(out_path,fn)) as f:
            json.dump(out_batch, f, indent = 2)

        time.sleep(5)
