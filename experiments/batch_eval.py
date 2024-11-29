from bert_score import BERTScorer
import os
from tqdm import tqdm
import json
import sys 
scorer = BERTScorer(model_type='bert-base-uncased')

if len(sys.argv) != 3:
        print("Usage: python batch_eval.py <mode> <model>")
        sys.exit(1)
try:
    mode = sys.argv[1]
    model = sys.argv[2]
except ValueError:
    print("Both arguments must be strings.")
    sys.exit(1)


in_path = f"./generation/{mode}/{model}"
out_path = f"./eval/{mode}/{model}"
os.makedirs(out_path, exist_ok=True)
fs = [f for f in os.listdir(in_path) if f.endswith('.json')]
fs = sorted(fs, key=lambda x: int(x.replace(".json","").split("_")[1]))
out = dict()

# Also called Jaccard coefficient
def exact_match_score(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = set1.intersection(set2)
    
    match_score = len(intersection) / len(set1.union(set2))
    return match_score

def pad_list(list_to_pad, longer_list, padding_value=""):
    padding_length = len(longer_list) - len(list_to_pad)
    if padding_length > 0:
        list_to_pad.extend([padding_value] * padding_length)
    return list_to_pad

precisions = []
recalls = []
f1s = []
ems = []

print("Start evaluating...")
for fn in tqdm(fs):
    new_data = []
    with open(os.path.join(in_path,fn),"r") as f:
        print(fn)
        item = json.load(f)
        for i in item:
            new_item = dict()

            predictions = list(set(i["output"]))
            references = list(set(i["reference"]))
            predictions, references = [x for x in predictions if x is not None], [x for x in references if x is not None] 
 
            em_score = exact_match_score(predictions, references) 
            if len(predictions) > len(references):
                references = pad_list(references, predictions)
            elif len(references) > len(predictions):
                predictions = pad_list(predictions, references)

            precision, recall, f1 = scorer.score(predictions, references)
            precision, recall, f1 = precision.tolist(), recall.tolist(), f1.tolist()
        

            new_item["id"] = i["qid"]
            new_item["label"] = i["label"]
            new_item["output"] = i["output"]
            new_item["reference"] = i["reference"]
            new_item["em_score"] = em_score
            ems.append(em_score)
            new_item["precision"] = precision
            precisions.append(sum(precision)/len(precision))
            new_item["recall"] = recall
            recalls.append(sum(recall)/len(recall))
            new_item["f1"] = f1
            f1s.append(sum(f1)/len(f1))
            new_data.append(new_item)
        
        
            # new_item["id"] = k
            # new_item["label"] = v["label"]
            # new_item["output"] = v["output"]
            # new_item["reference"] = v["reference"]
            # new_item["em_score"] = em_score
            # ems.append(em_score)
            # new_item["precision"] = result["precision"]
            # precisions.append(sum(result["precision"])/len(result["precision"]))
            # new_item["recall"] = result["recall"]
            # recalls.append(sum(result["recall"])/len(result["recall"]))
            # new_item["f1"] = result["f1"]
            # f1s.append(sum(result["f1"])/len(result["f1"]))
            # new_data.append(new_item)
        
    with open(os.path.join(out_path,fn),"w") as f:
        json.dump(new_data, f, indent=2)
    
p = sum(precisions) / len(precisions)
r = sum(recalls) / len(recalls)
f1 = sum(f1s) / len(f1s)
em = sum(ems) / len(ems)

with open(os.path.join(out_path,"summary"),"w") as f:
    f.write(f"precision: {p}\n")
    f.write(f"recall: {r}\n")
    f.write(f"f1: {f1}\n")
    f.write(f"em: {em}\n")


# Plot BERTscore heatmap
# plot_example(cands[0], refs[0], lang="en")
