import os
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import torch
import jieba


def load_model_and_tokenizer(args):
    config = AutoConfig.from_pretrained(args["model_dir"], trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args["model_dir"], trust_remote_code=True)
    model = AutoModel.from_pretrained(args["model_dir"], trust_remote_code=True).half().cuda()
    model = PeftModel.from_pretrained(model, os.path.join(args["save_dir"]), torch_dtype=torch.float32,
                                      trust_remote_code=True)
    model.half().cuda()
    model.eval()
    return tokenizer, model


def load_data(file_path):
    test_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            input_text = data["instruct"] + data["query"]
            output_text = data["answer"]
            test_data.append((input_text, output_text))
    return test_data


def evaluate_model(tokenizer, model, test_data):
    predictions = []
    references = []
    for inp, true_out in test_data:
        if not true_out.strip():
            continue

        response, _ = model.chat(tokenizer, inp, history=[])
        predictions.append(response)
        references.append(true_out)

    return predictions, references


def calculate_metrics(predictions, references):
    tokenized_preds_b = [list(jieba.cut(p)) for p in predictions]
    tokenized_refs_b = [[list(jieba.cut(r))] for r in references]

    bleu_score = corpus_bleu([[list(ref)] for ref in tokenized_refs_b], [list(pred) for pred in tokenized_preds_b])

    tokenized_preds_r = [' '.join(jieba.cut(pred)) for pred in predictions]
    tokenized_refs_r = [' '.join(jieba.cut(ref)) for ref in references]

    rouge = Rouge()
    rouge_scores = rouge.get_scores(tokenized_preds_r, tokenized_refs_r, avg=True)

    return bleu_score, rouge_scores


# Main execution
if __name__ == "__main__":
    data_name = "sleep"
    train_args_path = f"./checkpoint/{data_name}/train_deepspeed/result7/train_args.json"

    with open(train_args_path, "r") as fp:
        args = json.load(fp)

    tokenizer, model = load_model_and_tokenizer(args)
    file_path = 'data/sleep_train.txt'
    # file_path = 'data/sleep_test.txt'
    # file_path = 'data/sleep_dev.txt'
    test_data = load_data(file_path)
    predictions, references = evaluate_model(tokenizer, model, test_data)

    for i in range(min(5, len(predictions))):
        print(f"Prediction: {predictions[i]}")
        print(f"Reference: {references[i]}\n")

    bleu_score, rouge_scores = calculate_metrics(predictions, references)
    print(f"BLEU-4: {bleu_score}")
    print(f"ROUGE Scores: {rouge_scores}")
