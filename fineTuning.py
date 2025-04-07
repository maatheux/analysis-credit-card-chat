from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
import json
from Classes.Dataset import DatasetModel


def load_dataset():
    with open(r"./PreDatasets/CreditCard.json", "r", encoding="utf-8-sig") as f:
        dataset = json.load(f)
    
    return dataset


def data_processing(data):
    # inputs = tokenizer([x["chat"] for x in data], truncation=True, padding=True, max_length=512)
    # labels = tokenizer([x["result"] for x in data], truncation=True, padding=True, max_length=512)

    # return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels["input_ids"]}
    
    
    inputs = []
    labels = []
    
    for item in data:
        input_tokenized = tokenizer(item['chat'], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        label_tokenized = tokenizer(item['result'], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        
        inputs.append(input_tokenized["input_ids"].squeeze())
        labels.append(label_tokenized["input_ids"].squeeze())
        
    return {"input_ids": inputs, "labels": labels}


def train_model(model, dataset):
    training_args = TrainingArguments(
        output_dir="./Results/fine_result",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Acumula gradientes para simular batch maior
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        fp16=True,
        optim="adafactor",  # Otimizador mais leve que AdamW
        lr_scheduler_type="cosine",  # Agendador de learning rate eficiente
        warmup_steps=100,
        report_to="none"  # Desativa logs desnecessários
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    trainer.train()
    
    model.save_pretrained("./Models/trained_model")
    tokenizer.save_pretrained("./Models/trained_model")




if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B") # .to("cpu") ou pode ser outro arg: device_map = "auto" -> forcar ou escolher auto
    
    model = model.to("cuda") # manda modelo para GPU
    
    tokenizer.pad_token = tokenizer.eos_token  # token de fim de texto, pois o modelo nao possui um token padding padrao
    
    dataset = load_dataset()
    
    tokenized_data = data_processing(dataset)
    
    final_dataset = DatasetModel(tokenized_data, tokenizer.pad_token_id)

    train_model(model, final_dataset)
    



