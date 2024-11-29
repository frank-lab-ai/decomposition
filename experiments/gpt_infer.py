from openai import OpenAI
import openai
import os
import json 
import sys
from template import Template
from tqdm import tqdm
import time
import pprint


openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=openai.api_key)

with open('./data/test.json', 'r') as f:
    data = json.load(f)

if len(sys.argv) != 2:
        print("Usage: python gpt_infer.py <mode>")
        sys.exit(1)
try:
    mode = sys.argv[1]
except ValueError:
    print("Argument must be strings.")
    sys.exit(1)

model_name = "gpt_4o"
all_inputs = [Template(item["label"], item["desc"].rstrip("."), mode=mode).template for item in data]


BATCH_SIZE = 10

out_path = f"./generation/{mode}/{model_name}"
os.makedirs(out_path, exist_ok=True)

batchfile_path = f"data/{mode}_batch.jsonl"
def write_batch_file (inputs):
     with open(batchfile_path, "w") as f:
          for i, item in enumerate(inputs):
               item = item.replace('"', '\\"')
               item = item.replace('\n', '\\n')
               f.write(f'{{"custom_id": "request-{i+1}", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "gpt-4o", "messages": [{{"role": "user", "content": "{item}"}}],"max_tokens": 1000}}}}\n')

# write_batch_file(all_inputs)     

def call_gpt_batch_api (fname):
    # batch_input_file = client.files.create(
    #     file=open(fname, "rb"),
    #     purpose="batch")
    
    # batch_input_file_id = batch_input_file.id

    # response = client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    #     metadata={
    #     "description": "nightly infer job"}
    # )

    # print(response)
    # while client.batches.retrieve(response.id).status != 'completed':
    #     time.sleep(5)
    #     print(client.batches.retrieve(response.id).status)
    #     if client.batches.retrieve(response.id).status == 'failed':
    #         print(client.batches.retrieve(response.id))
    #         return 

    # file_response = client.files.content(client.batches.retrieve(response.id).output_file_id)
    # result = file_response.content

    result_file_name = f"data/{mode}_batch_job_results.jsonl"
    # with open(result_file_name, 'wb') as file:
    #     file.write(result)
    results = []
    with open(result_file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)
    results = [r['response']['body']['choices'][0]['message']['content'] for r in results]

    out = []
    for i in range(len(all_inputs)):
        item_result = dict()
        item_result["qid"] = data[i]["qid"]
        item_result["label"] = data[i]["label"]
        if "Parts: " in results[i]:
            start_pos = results[i].find("Parts: ") + len("Parts: ")
            item_result["output"] = results[i][start_pos:].split(", ")
        else:
            item_result["output"] = results[i].split(", ")
        item_result["reference"] = [p["plabel"] for p in data[i]["part"]]
        out.append(item_result)
    
    BATCH_SIZE = 10
    for it in tqdm(range (0, len(all_inputs), BATCH_SIZE)):
        out_batch = out[it:it+BATCH_SIZE]
        fn = f'test_{it//BATCH_SIZE+1}.json'
        with open(os.path.join(out_path,fn), 'w') as f:
            json.dump(out_batch, f, indent = 2)
    
call_gpt_batch_api(batchfile_path)




# def call_gpt_api_in_batches(all_inputs, model="gpt-4o", batch_size=BATCH_SIZE):
#     result = {}

#     for i in tqdm(range(0, len(all_inputs), batch_size)):
#         batch_prompts = all_inputs[i:i+batch_size]
#         print("inputs:", batch_prompts)

#         completion = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=batch_prompts
#         )
#         print(completion)
#         print(completion.choices[0].message)

        # stream = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=batch_prompts,
        #     stream=True,)

        # output = []
        # for chunk in stream:
        #     if chunk.choices[0].delta.content is not None:
        #         output.append(chunk.choices[0].delta.content)
        #         print("chunk:", chunk.choices[0].delta.content, end="")

        # print("output:", output)
        # for j in range(len(batch_prompts)):
        #     item_result = dict()
        #     item_result["label"] = data[i+j]["label"]
        #     item_result["output"] = output[i].split(", ")
        #     item_result["reference"] = [p["plabel"] for p in data[i+j]["part"]]
        #     result[data[i+j]["qid"]] = item_result
        

        # with open(f'./generation/{mode}/{model_name}/test_{i//batch_size+1}.json', 'w') as f:
        #     json.dump(result, f, indent = 2)


# call_gpt_api_in_batches(all_inputs)

