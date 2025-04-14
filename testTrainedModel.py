from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def result_preview(question, generator):
    prompt = f"Retorne um resultado com apenas 'Sim' ou 'Não' para a pergunta que terá sentido se o cliente possui um cartão de crédito ou não\nChat: {question}\nResultado:"
    answer = generator(
        prompt,
        max_new_tokens=2,
        num_return_sequences=1,
        temperature=0.01
    )
    
    return answer[0]["generated_text"].split("Resultado:")[-1].strip()


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("./Models/trained_model", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("./Models/trained_model")
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    chat = "Atendente: Possui o cartão do nosso banco?\nCliente: Acredito que o cartão que eu tenha aqui seja de um outro banco"
    print(result_preview(chat, generator))
