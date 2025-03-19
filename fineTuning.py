from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
import json


def load_dataset():
    with open(r"./PreDatasets/CreditCard.json", "r", encoding="utf-8-sig") as f:
        dataset = json.load(f)
    
    return dataset


# def data_processing(data):
#     inputs = tokenizer([x["chat"] for x in data], truncation=True, padding=True, max_length=512)
#     labels = tokenizer([x["result"] for x in data], truncation=True, padding=True, max_length=512)
# 
#     return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels["input_ids"]}




if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B") 
    
    # dataset = load_dataset()
    
    



