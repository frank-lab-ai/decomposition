# Object and Concept Decomposition Using Neuro-symbolic Methods



## Setup

The repo uses Python 3.8.10 with packages listed in `requirement.txt`. 

Create a Python virtual environment under the root folder and activate it such as this:

``` bash
python3.x -m venv env
source env/bin/activate
pip install -r requirements.txt
```

To run inference using OpenAI, store the API key as an environment variable `OPENAI_API_KEY`. To run inference using Groq, store it as `GROQ_API_KEY`. 



## Data

The data splits as well as few-shot prompt prefix used in the experiments is in `data`. The data split is:

| Split      | Items  |
| ---------- | ------ |
| Train      | 11,919 |
| Validation | 1,489  |
| Test       | 1,491  |



## Experiments

The paper lists four experiment modes (for exact mode name, check `template.py`) and three models (`llama-3.1`, `gpt4o` and `t5`). To generate decomposition:

```bash
# Generate decomposition
# T5
python experiments/batch_infer.py <mode>
# GPT4o
python experiments/gpt_infer.py <mode>
# Llama
python experiments/llama_infer.py <mode>
```

After LM generated decompositions, they are stored in `generation` under `mode/model` folder. An example file is as follow:

```
[
  {
    "qid": "Q5248631",
    "label": "surf culture",
    "output": [
      "surfboard",
      "surfer",
      "surfing competition",
      "wave",
      "beach",
      "surf fashion",
      "surfing community",
      "surf magazine"
    ],
    "reference": [
      "surf music",
      "Surfwear"
    ]
   },
   ...
]
```

It contains the **QID** of Wikidata item (its full data accessible at `https://www.wikidata.org/wiki/{QID}`), its label, LM generated decomposition **output** and the **reference** decomposition from WikiDS dataset. 



To evaluate that experiment mode and model:

```bash
# Evaluate decomposition
python experiments/batch_eval.py <mode> <model>
```

This will output per-batch statistics in JSON format and a summary file `summary` for aggregated BertScores (Precision, Recall, F1) and exact match.

To finetune T5 model,

```bash
# Finetune T5 model
python experiments/t5_finetune.py
```

