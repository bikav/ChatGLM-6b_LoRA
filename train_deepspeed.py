import os
import json
import torch
import deepspeed

from pprint import pprint
from torch.utils.data import RandomSampler, DataLoader
from dataset import load_data, NerCollate
from config_utils import ConfigParser
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

from peft import (
    LoraConfig,
    get_peft_model,
)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def main():
    args = {
        "data_name": "sleep",
        "model_dir": "ChatGLM-6b",
        "lora_r": 32,
        "max_source_length": 128,
        "max_target_length": 64,
        "instruct_column": "instruct",
        "query_column": "query",
        "response_column": "answer",
        "train_path": "data/sleep_train.txt",
        "dev_path": "data/sleep_dev.txt",
        "ignore_pad_token_for_loss": True,
        "train_batch_size": 6,
        "gradient_accumulation_steps": 2,
        "save_dir": "./lora_model",
        "num_train_epochs": 20,
        "local_rank": -1,
        "log_steps": 10,
        "save_steps": 50,  # 每50步保存一次
    }

    loss_values = []

    config_parser = ConfigParser(args)
    args = config_parser.parse_main()

    pprint(vars(args))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, "train_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

    model = get_peft_model(model, config)
    model = model.cuda()

    conf = {
        "train_micro_batch_size_per_gpu": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-5, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 5e-4}
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "steps_per_print": args.log_steps
    }

    print_trainable_parameters(model)

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    train_data = load_data(args.train_path)
    ner_collate = NerCollate(args, tokenizer)
    train_dataloader = DataLoader(
        train_data,
        batch_size=conf["train_micro_batch_size_per_gpu"],
        sampler=RandomSampler(train_data),
        drop_last=True,
        collate_fn=ner_collate.collate_fn
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf, model=model, model_parameters=model.parameters())
    model_engine.train()

    total_step = int(len(train_dataloader) * args.num_train_epochs / conf["gradient_accumulation_steps"])
    global_step = 0

    for i_epoch in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            outputs = model_engine.forward(input_ids=input_ids, labels=labels)
            loss = outputs[0]
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1

            loss_values.append(loss.item())

            if global_step % args.log_steps == 0:
                print("loss:{}, global_step:{}/{}".format(float(loss.item()), global_step, total_step))

            # 每隔50步保存一次模型
            if global_step % 50 == 0:
                save_path = os.path.join(args.save_dir, f"model_step_{global_step}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model_engine.save_pretrained(save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('./loss_curve.png')  # 保存loss曲线到文件
    plt.close()


if __name__ == "__main__":
    main()
