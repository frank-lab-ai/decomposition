import os
import json
class Template:
    def __init__(self, item, context="", mode="zero-shot"):
        self.mode = mode
        self.template = ""
        if self.mode == "zero-shot":
             self.template = f"list all possible parts of the {item}, separated by commas:"
        elif self.mode == "gpt-zero-shot":
             self.template = f"list all possible parts that make up the {item}. Reply in this format: {item} includes: [part], [part], ..."
        elif self.mode == "zero-shot-context":
             self.template = f"{item} is the {context}.\n List all possible parts of the {item}, separated by commas:"
        elif self.mode == 'gpt-zero-shot-context':
             self.template = f"{item} is the {context}. List all possible parts that make up the {item}. Reply in this format: {item} includes: [part], [part], ..."
        elif self.mode == "few-shot-context":
             if os.path.exists("data/prefix"):
                 with open("data/prefix","r") as f:
                     prefix = "".join(f.readlines())
             else:
                 prefix = self.make_few_shot_prefix()
             self.template = f"{prefix}\nNow you try:\nItem: {item}\nContext: {context}\nParts: "
        elif self.mode == "finetuned-zero-shot-context":
             self.template = f"decompose item: {item}.\t context: {context}"
        else:
            pass

    @staticmethod
    def make_few_shot_prefix():
        prefix = ""
        with open("data/in_context_held_out.json","r") as f:
            data = json.load(f)
        for i in data:
            item = i["label"]
            context = i["desc"]
            temp = i["part"]
            temp = [i["plabel"] for i in temp]
            parts = ", ".join(temp)
            prefix += f"Item: {item}\nContext: {context}\nParts: {parts}\n\n"
        with open("data/prefix","w") as f:
            f.write(prefix)
        return prefix 
    