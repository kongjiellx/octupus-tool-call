from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# qwen2.5-7b pretrain model path
model_path = "/data/.modelcache/common-crawl-data/model-repo/Qwen/Qwen2.5-7B/d149729398750b98c0af14eb82c78cfe92750796"
save_path = "your/save/path"

old_tokenizer = AutoTokenizer.from_pretrained(model_path)
new_tokenizer = AutoTokenizer.from_pretrained(model_path)


new_tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "<|SYSTEM|>",
        "<|USER|>", 
        "<|ASSISTANT|>",
        "<|FUNCTION_CALL|>",
        "<|PARAMETERS|>",
        "<|FUNCTION_OUTPUT|>",
        "<|CONTENT|>"
    ]
}, replace_additional_special_tokens=False)
new_tokenizer.save_pretrained(save_path)

device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)
print(model.get_input_embeddings().weight.shape)
print(model.get_output_embeddings().weight.shape)

new_tokens = {
    "<|SYSTEM|>": "system", 
    "<|USER|>": "user", 
    "<|ASSISTANT|>": "assistant", 
    "<|FUNCTION_OUTPUT|>": "tool response", 
    "<|FUNCTION_CALL|>": "tool call", 
    "<|PARAMETERS|>": "arguments", 
    "<|CONTENT|>": "content"
}

for new_token, text in new_tokens.items():
    old_tokens = old_tokenizer.encode(text)
    print(new_token, old_tokens)

    old_token_embeddings = model.get_input_embeddings()(torch.tensor(old_tokens))
    new_embedding = old_token_embeddings.mean(dim=0)
    model.get_input_embeddings().weight.data[new_tokenizer.get_vocab()[new_token]] = new_embedding
    
    old_output_embeddings = model.get_output_embeddings().weight[old_tokens, :]
    new_output_embedding = old_output_embeddings.mean(dim=0)
    model.get_output_embeddings().weight.data[new_tokenizer.get_vocab()[new_token]] = new_output_embedding

print("save")
model.save_pretrained(save_path)