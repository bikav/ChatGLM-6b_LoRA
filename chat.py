import os
import torch
import json
from pprint import pprint
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

data_name = "sleep"

train_args_path = "lora_model/train_args.json".format(data_name)
with open(train_args_path, "r") as fp:
    args = json.load(fp)


config = AutoConfig.from_pretrained(args["model_dir"], trust_remote_code=True)
pprint(config)
tokenizer = AutoTokenizer.from_pretrained(args["model_dir"],  trust_remote_code=True)

model = AutoModel.from_pretrained(args["model_dir"],  trust_remote_code=True).half().cuda()
model = model.eval()
model = PeftModel.from_pretrained(model, os.path.join(args["save_dir"]), torch_dtype=torch.float32, trust_remote_code=True)

model.half().cuda()
model.eval()

while True:
    inp = input("User >>> ")
    if not isinstance(inp, str):
        inp = str(inp)
    response, history = model.chat(tokenizer, inp, history=[])
    print("Chat >>> ", response)
    print("="*100)
